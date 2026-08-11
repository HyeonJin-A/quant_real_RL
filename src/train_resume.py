"""
V9 PPO 이어학습(resume) 전용 스크립트 (`V9 Design TODO - Resume 학습 지원.md` 참고).

기존 소스(`train.py`/`env.py`/`eval.py`/`algorithm.py`/`prep_features.py`)는 전혀 수정하지
않는다 — resume에 필요한 로직은 전부 이 파일에만 있고, `train.py`/`eval.py`의 재사용 가능한
함수·상수는 읽기 전용으로 import만 한다 (2026-07-31 설계 결정: fresh-run 경로가 이미
검증되어 신뢰받고 있고 장시간 학습 런이 자주 도는 환경이라, resume이라는 드물게 쓰이는
실험적 경로 때문에 기존 경로를 건드릴 위험을 없애기 위함).

핵심 설계:
- T_end(--timesteps) / T_ref(--schedule-ref-total) 분리: lr은 새 종료 스텝(T_end)까지
  다시 감쇠시키고, ent_coef/explore_bonus는 원 런 기준 총스텝(T_ref)에서 이미 끝난
  스케줄을 그대로 유지한다.
- lr piecewise 스케줄: --lr-knee-step 이전 구간은 원 스케줄(LR_START*(1-t/T_ref))을
  그대로 재현하고, 그 지점의 lr을 기준으로 --timesteps까지 다시 선형 0-수렴시킨다.
- `build_callbacks()`는 train.py 원본을 그대로 import해 쓰되, args.timesteps만
  T_ref로 바꿔치기한 얕은 복사본을 넘겨서 ent_coef/explore_bonus가 원 런 기준
  스케줄을 유지하도록 한다 (train.py의 args.timesteps 참조 3곳이 전부
  build_callbacks() 내부에만 있다는 점을 이용 — 소스 수정 불필요).

필수 인자는 `--resume-from`/`--seed-best-info` 둘뿐 — 나머지는 전부 기본값이 있음
(2026-07-31: `--leverage`/`--episode-days`/`--stride`/`--n-steps`/`--batch-size`/
`--workers`/`--symbols`/`--timesteps`/`--schedule-ref-total`/`--lr-knee-step`의 기본값은
`models/best/v9_maskablerl_seed0_0728-1924_best.zip` 베이스라인 기준으로 고정한 것 —
**다른 체크포인트를 resume할 땐 그 런의 실제 설정으로 반드시 override할 것**, 특히
leverage/episode_days/stride/n_steps/batch_size/workers/symbols는 원 체크포인트와
어긋나면 obs shape은 그대로라 에러 없이 조용히 잘못 학습됨).

사용법 (0728-1924 베이스라인 resume, 최소):
  python src/train_resume.py \\
    --resume-from models/best/v9_maskablerl_seed0_0728-1924_best.zip \\
    --seed-best-info models/best/v9_maskablerl_seed0_0728-1924_best_info.json

다른 체크포인트/설정으로 override하는 예:
  python src/train_resume.py \\
    --resume-from PATH --seed-best-info PATH \\
    --timesteps 300000000 --schedule-ref-total 100000000 --lr-knee-step 90000000 \\
    --leverage 3.0 --episode-days 30 --stride 1 --n-steps 2048 --batch-size 512 \\
    --workers 14 --symbols BTC-USDT-SWAP ETH-USDT-SWAP
"""
import os
import re
import sys
import json
import copy
import argparse
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))
import train as _train_mod  # noqa: E402  # MODEL_DIR/MODE_SUBDIR를 런타임에 패치하기 위해 모듈 자체를 참조
from train import ROOT_DIR, MODEL_DIR, LOG_DIR, LR_START, make_env_fn, build_callbacks  # noqa: E402
from eval import cache_path_for, split_bounds, load_feature_metadata  # noqa: E402
from env import FEATURE_NAMES  # noqa: E402


def _orig_run_name(resume_from):
    """resume_from 경로에서 원본 run_name 추출 (models|logs/resume/{원본}/ 그룹핑용).
    '_best'/'_final'/'_<N>_steps' 접미사와 확장자를 제거해 같은 원본 체크포인트에서
    나온 여러 resume 시도가 같은 폴더 아래 모이게 한다."""
    base = os.path.splitext(os.path.basename(resume_from))[0]
    for suffix in ("_best", "_final"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    m = re.match(r"(.+)_\d+_steps$", base)
    return m.group(1) if m else base


def make_lr_fn(lr_start, t_ref, t_end, knee_step=None, knee_value=None):
    """progress_remaining -> lr.

    knee_step 이전(절대 스텝 기준): 원 스케줄 LR_START*(1 - t/t_ref) 그대로 재현.
    knee_step 이후: knee 지점의 lr(knee_value, 미지정 시 위 식으로 자동 산출)에서
    t_end까지 선형으로 0에 수렴.

    knee_step이 None이면 LR_START*progress_remaining 단일 선형(= t_end 기준 처음부터
    다시 감쇠) — 대부분의 resume 시나리오에선 이미 많이 감쇠된 체크포인트에 이 식을
    쓰면 lr이 큰 폭으로 튀어오르므로 지정하는 것을 강력히 권장한다 (호출부에서 경고 출력).
    """
    if knee_step is None:
        return lambda pr: lr_start * pr

    lr_knee = knee_value if knee_value is not None else lr_start * (1.0 - knee_step / t_ref)

    def lr_fn(progress_remaining):
        t = (1.0 - progress_remaining) * t_end
        if t < knee_step:
            return lr_start * (1.0 - t / t_ref)
        return lr_knee * max(0.0, (t_end - t) / max(t_end - knee_step, 1))

    return lr_fn


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    # --- resume 전용 인자 ---
    parser.add_argument("--resume-from", required=True, help="이어학습할 체크포인트 zip 경로")
    parser.add_argument("--timesteps", type=int, default=300_000_000,
                        help="최종 누적 절대 스텝(T_end, 기본 300M). 체크포인트의 num_timesteps보다 커야 함")
    parser.add_argument("--schedule-ref-total", type=int, default=100_000_000,
                        help="ent_coef/explore_bonus의 *_frac 기준 총스텝(T_ref, 기본 100M — 원 런의 "
                             "--timesteps 기본값). 다른 원 런을 resume할 땐 그 런의 실제 --timesteps로 변경")
    parser.add_argument("--lr-knee-step", type=int, default=90_000_000,
                        help="lr 무릎 지점(절대 스텝, 기본 90M — 0728-1924 베이스라인 기준). "
                             "None으로 주면(코드상 미지원, 값을 직접 바꿀 것) LR_START*progress_remaining "
                             "단일 선형(t_end 기준)이 되어 대부분의 resume에서 lr이 급등하니 주의")
    parser.add_argument("--lr-knee-value", type=float, default=None,
                        help="무릎 지점 lr. 미지정 시 LR_START*(1-knee/ref)로 자동 산출")
    parser.add_argument("--resume-reset-best", action="store_true",
                        help="best_score/kpi_history를 이어받지 않고 -inf/빈 상태로 리셋")
    parser.add_argument("--seed-best-info", required=True,
                        help="원 체크포인트의 _best_info.json 경로. btc_v10_kpi 원시값을 그대로 "
                             "best_score/kpi_history 시드값으로 사용 (v10_kpi는 거래단위 t-통계량이라 "
                             "v9_kpi 시절과 달리 조정 가능한 가중치가 없어 재계산 불필요 — 2026-08-06). "
                             "v10_kpi 이전 포맷 파일(btc_v10_kpi 필드 없음)은 시드 불가, "
                             "--resume-reset-best를 쓸 것")

    # --- 원 체크포인트와 반드시 일치해야 하는 값. 기본값은 0728-1924 베이스라인
    #     (models/best/v9_maskablerl_seed0_0728-1924_best.zip) 기준으로 고정 —
    #     다른 체크포인트를 resume할 땐 아래 값들을 그 런의 실제 설정으로 반드시 바꿔서 지정할 것
    #     (leverage=3.0은 quant_main의 「RL 실전-백테스트 괴리 점검」 문서와 커밋 e83032f의 코드
    #     해석(leverage=None+exit_mode=rl -> 3.0)으로 교차 확인함, 2026-07-31).
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    parser.add_argument("--leverage", type=float, default=3.0,
                        help="원 런과 반드시 동일해야 함 — liq_dist 관측 피처가 leverage에 의존")
    parser.add_argument("--episode-days", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=14,
                        help="원 런과 다르면 롤아웃 크기(n_steps*n_envs)가 달라져 업데이트 semantics가 바뀜")

    # --- 나머지: train.py와 동일 기본값 유지 ---
    parser.add_argument("--final-split", action="store_true")
    parser.add_argument("--main-core", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--kpi-smooth-window", type=int, default=3)
    parser.add_argument("--eval-segments", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--ent-coef-start", type=float, default=0.01)
    parser.add_argument("--ent-coef-end", type=float, default=0.005)
    parser.add_argument("--ent-coef-hold-frac", type=float, default=0.7)
    parser.add_argument("--explore-bonus-start", type=float, default=0.005)
    parser.add_argument("--explore-bonus-decay-frac", type=float, default=0.5)
    parser.add_argument("--cache-suffix", default="")
    parser.add_argument("--dummy-vec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-features", nargs="*", default=None,
                        help="관측에서 제외할 피처 이름(공백 구분, env.FEATURE_NAMES 중). "
                             "미지정 시 --resume-from 옆 *_features.json 메타데이터에서 자동 로드 "
                             "(원 체크포인트와 다른 exclude_features로 resume하면 관측 차원이 어긋나 "
                             "MaskablePPO.load()에서 즉시 에러가 남 — 2026-08-09)")
    args = parser.parse_args()

    if args.exclude_features is not None:
        bad_features = sorted(set(args.exclude_features) - set(FEATURE_NAMES))
        if bad_features:
            raise SystemExit(f"--exclude-features: 알 수 없는 이름 {bad_features}; 허용: {list(FEATURE_NAMES)}")
        exclude_features = args.exclude_features
    else:
        meta = load_feature_metadata(args.resume_from)
        if meta is not None:
            exclude_features = meta["exclude_features"]
            print(f"[resume] exclude_features 자동 로드({args.resume_from} 옆 메타데이터): {exclude_features}")
        else:
            exclude_features = []
            print(f"[resume] {args.resume_from} 옆에 *_features.json 없음 — 구 체크포인트로 간주, 전체 18피처 사용")

    env_kwargs = {"leverage": args.leverage, "exclude_features": tuple(sorted(set(exclude_features)))}

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {args.main_core})
            num_cpus = os.cpu_count() or 1
            if args.dummy_vec:
                print(f"[CPU Affinity] Main process -> Core {args.main_core} | DummyVecEnv: {args.workers} envs 전부 메인에서 순차 실행")
            else:
                print(f"[CPU Affinity] Main process -> Core {args.main_core} | {args.workers} workers -> Cores 1~{min(args.workers, num_cpus - 1) if num_cpus > 1 else 0}")
        except Exception as e:
            print(f"[CPU Affinity] Failed to set affinity: {e}")

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.logger import configure as configure_sb3_logger

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    cache_paths = {s: cache_path_for(s, args.cache_suffix) for s in args.symbols}
    for s, p in cache_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"cache not found for {s}: {p} — run prep_features_v9.py first")
    bounds = split_bounds(list(cache_paths.values()), final=args.final_split)

    episode_len_rows = args.episode_days * 1440 // args.stride * args.stride
    env_fns = []
    symbols = list(cache_paths.keys())
    for w in range(args.workers):
        sym = symbols[w % len(symbols)]
        path = cache_paths[sym]
        lo, hi = bounds[path]["train"]
        env_fns.append(make_env_fn(path, lo, hi, episode_len_rows, args.stride, args.seed * 1000 + w, env_kwargs,
                                   worker_idx=w, pin_core=not args.dummy_vec))

    vec_env = DummyVecEnv(env_fns) if args.dummy_vec else SubprocVecEnv(env_fns, start_method="spawn")

    run_name = f"v10_maskablerl_seed{args.seed}_{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%m%d-%H%M')}_resume"
    print(f"run_name: {run_name}")

    # 저장 경로: models|logs/resume/{원본 체크포인트 run_name}/{이번 resume run_name}/...
    # 같은 원본에서 나온 여러 resume 시도를 한곳에 모으고, v9_maskablerl/ 트리와는 분리한다.
    orig_name = _orig_run_name(args.resume_from)
    resume_model_root = os.path.join(MODEL_DIR, "resume", orig_name)
    resume_log_root = os.path.join(LOG_DIR, "resume", orig_name)
    # build_callbacks()는 train.py 원본을 그대로 import해 쓰므로 내부에서
    # os.path.join(MODEL_DIR, MODE_SUBDIR, run_name)을 직접 계산한다 — 그 결과도 위 경로로
    # 향하도록 train 모듈 자체의 전역값을 런타임에 패치한다 (파일은 안 건드림, 이 프로세스
    # 안에서만 유효). train.py에서 이 두 상수를 쓰는 곳은 build_callbacks()뿐이라 안전함
    # (main()은 우리가 호출하지 않으므로 그쪽 참조는 영향 없음).
    _train_mod.MODEL_DIR = resume_model_root
    _train_mod.MODE_SUBDIR = ""

    if args.lr_knee_step is None:
        print("[resume] ⚠️  --lr-knee-step 미지정 — lr이 LR_START*progress_remaining(t_end 기준) "
              "단일 선형으로 재계산되어, 이미 감쇠된 체크포인트에선 lr이 큰 폭으로 튀어오를 수 있음. "
              "의도한 게 아니라면 --lr-knee-step을 지정할 것.")
    lr_fn = make_lr_fn(LR_START, args.schedule_ref_total, args.timesteps,
                       args.lr_knee_step, args.lr_knee_value)

    model = MaskablePPO.load(
        args.resume_from,
        env=vec_env,
        device="cpu",
        custom_objects={"learning_rate": lr_fn, "lr_schedule": lr_fn},
        seed=args.seed,
        tensorboard_log=LOG_DIR,
    )
    start_step = model.num_timesteps
    delta = args.timesteps - start_step
    if delta <= 0:
        raise ValueError(f"--timesteps({args.timesteps:,}) <= 체크포인트 num_timesteps({start_step:,})")
    resume_start_lr = lr_fn(1.0 - start_step / args.timesteps)
    print(f"[resume] loaded from {args.resume_from} | num_timesteps={start_step:,} | "
          f"delta={delta:,} (-> T_end={args.timesteps:,}) | lr@resume={resume_start_lr:.3e}")

    model_dir = os.path.join(resume_model_root, run_name)
    os.makedirs(model_dir, exist_ok=True)
    runmeta = {
        "resume_from": os.path.abspath(args.resume_from),
        "resume_start_step": start_step,
        "git_commit": _git_commit(),
        "args": {k: v for k, v in vars(args).items()},
    }
    with open(os.path.join(model_dir, f"{run_name}_runmeta.json"), "w", encoding="utf-8") as f:
        json.dump(runmeta, f, indent=2, ensure_ascii=False)

    model.set_logger(configure_sb3_logger(os.path.join(resume_log_root, run_name), ["stdout", "tensorboard"]))

    schedule_args = copy.copy(args)
    schedule_args.timesteps = args.schedule_ref_total  # build_callbacks 내부 ent_coef/explore_bonus만 T_ref 기준
    callbacks = build_callbacks(schedule_args, cache_paths, bounds, run_name, env_kwargs)

    validation_cb = next(cb for cb in callbacks if type(cb).__name__ == "ValidationCallback")
    if args.resume_reset_best:
        print("[resume] --resume-reset-best: best_score/kpi_history를 -inf/빈 상태로 시작")
    elif args.seed_best_info:
        with open(args.seed_best_info, encoding="utf-8") as f:
            info = json.load(f)
        if "btc_v10_kpi" not in info:
            raise SystemExit(f"[resume] ⚠️  {args.seed_best_info}는 v10_kpi 이전 포맷이라 시드 불가 "
                              "(btc_v10_kpi 필드 없음). --resume-reset-best를 쓰거나 v10_kpi로 "
                              "재평가한 정보 파일을 지정하세요.")
        seed_kpi = info["btc_v10_kpi"]
        validation_cb.best_score = seed_kpi
        validation_cb.kpi_history.append(seed_kpi)
        print(f"[resume] best_score seeded from {args.seed_best_info}: {seed_kpi:+.4f} "
              f"(원 파일의 btc_v10_kpi 원시값 그대로 사용 — kpi_history엔 이 값 1개만 시드, 나머지는 복구 불가)")
    else:
        print("[resume] ⚠️  --seed-best-info 미지정 — best_score가 -inf에서 시작해 첫 합격 검증이 "
              "무조건 best로 저장됨. 원 체크포인트보다 못한 모델이 best를 덮어쓸 수 있음.")

    model.learn(total_timesteps=delta, callback=callbacks, reset_num_timesteps=False)

    final_path = os.path.join(model_dir, f"{run_name}_final")
    model.save(final_path)
    print(f"final model saved: {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
