# V9 Design TODO — Resume(이어학습) 지원

상태: **구현 완료** (2026-07-31, `src/train_resume.py`) — 스모크 테스트(`_recent120d`
캐시, 소규모 timesteps: 프레시 20k 학습 → resume 24.5k 추가 → 45k)로 로드/피스와이즈
lr/T_ref·T_end 분리/즉시 검증/best_score 시드/runmeta 기록까지 전체 경로 동작 확인.
`git diff src/train.py src/env.py src/eval.py src/algorithm.py src/prep_features.py`는
빈 결과 (기존 소스 무변경 확인).

## 0. 배경 / 문제

베이스라인(`models/best/v9_maskablerl_seed0_0728-1924_best.zip`, 96,001,920 스텝)의 valid 지표가
100M 근처까지도 상승 곡선을 그리고 있어 더 학습하면 개선 여지가 있어 보임. 그런데
`src/train.py`의 lr 스케줄이 `--timesteps`(기본 100M) 시점에 정확히 0으로 수렴하도록
짜여 있어(`learning_rate=lambda progress_remaining: LR_START * progress_remaining`),
그대로는 100M 이후 추가 학습이 사실상 무의미함 (lr=0).

또한 `src/train.py`에는 저장된 체크포인트를 불러와 학습을 재개하는 코드 경로 자체가
없음 (`MaskablePPO.load()`는 `eval.py`의 추론 전용 경로에만 존재). `--final-split` 인자
설명(train.py:276~278)에 이미 "기존 체크포인트는 lr/ent_coef/explore_bonus 스케줄이
소진돼 있어 이어붙여도 사실상 안 배움"이라 명시돼 있었던 것도 바로 이 문제.

## 1. 목표

- 기존 체크포인트(가중치 + optimizer 상태 + num_timesteps)를 로드해 이어서 학습.
- lr 스케줄을 "이어학습 시점의 값"에서 시작해 새 종료 스텝까지 다시 감쇠시키는
  piecewise 스케줄로 재정의.
- ent_coef / explore_bonus는 기존처럼 절대 스텝 기준 스케줄 종료 후 값 유지.
- **기존 소스(`src/train.py`, `src/env.py`, `src/eval.py`, `src/algorithm.py`,
  `src/prep_features.py`) 전부 한 글자도 수정하지 않는다.** resume 전용 로직은 신규
  파일 `src/train_resume.py`에 전부 담고, 기존 모듈에서 재사용 가능한 함수/클래스는
  **읽기 전용으로 import만** 한다 (2026-07-31 결정, 07-31 수정 — 처음엔 `train.py`
  국한으로 좁게 잡았으나, 라이브 학습 런이 자주 도는 환경에서 신뢰도가 걸린 건
  `train.py`뿐 아니라 `env.py`(보상/관측 정의)·`eval.py`(검증/평가 지표) 등 기존
  코드 전체이므로 "이미 검증된 기존 소스는 무엇이든 건드리지 않는다"로 원칙을
  확장. resume이라는 드물게 쓰이는 실험적 경로 때문에 이미 신뢰받는 어떤 경로도
  건드릴 위험을 아예 없애는 쪽을 택함).

## 2. SB3 내부 동작 확인 사항 (근거)

`venv/.../stable_baselines3/common/base_class.py`, `on_policy_algorithm.py` 확인:

- **자동 복원되는 것**: 정책 가중치, Adam optimizer 상태(`set_parameters`, base_class.py:744),
  `num_timesteps`(저장 데이터 포함), obs/action space 검증(`check_for_correct_spaces`).
  이 프로젝트는 `VecNormalize`를 쓰지 않으므로(DummyVecEnv만 사용) resume 시 최대
  함정인 관측 정규화 통계 불일치 문제는 애초에 발생하지 않음.
- **복원 안 되는 것**: lr 스케줄 함수, `ValidationCallback`의 `best_score`/`kpi_history`/
  `last_eval` (콜백 상태는 zip에 없음), 학습 시 env kwargs(leverage 등, zip에 기록 안 됨).
- **`_setup_model()`이 항상 `learning_rate`로부터 `lr_schedule`을 재생성**
  (`_setup_lr_schedule` → `self.lr_schedule = FloatSchedule(self.learning_rate)`,
  base_class.py:274-276). 즉 `load()`의 `custom_objects`에 `lr_schedule`만 넣으면
  **무시되고 옛 스케줄로 조용히 학습됨** — 반드시 `learning_rate` 자체를 오버라이드해야 함.
- **`reset_num_timesteps=False`일 때 `learn(total_timesteps=N)`은 "추가분"이 아니라
  `total_timesteps += self.num_timesteps`로 가산됨** (base_class.py:416). 즉 이미 96M을
  가진 모델에 `total_timesteps=300_000_000`을 그대로 넘기면 396M까지 돌아버림.
  → `--timesteps`는 "최종 누적 절대 스텝"으로 의미를 고정하고, 내부에서
  `delta = args.timesteps - model.num_timesteps`를 계산해 `learn()`에 넘겨야 함.
- lr 계산 시점(`on_policy_algorithm.py:330`, `train()` 호출 직전)에 `total_timesteps`는
  절대 총스텝이므로 `t = (1 - progress_remaining) * t_end`로 절대 스텝을 복원 가능.

## 3. 설계

### 3-0. 파일 분리 전략 — `src/train_resume.py` 신설

`train.py`를 전혀 고치지 않고도 resume이 가능한 근거: `args.timesteps`를 참조하는
곳이 전체 `train.py`에 정확히 3곳뿐이고(103, 107, 262행), 셋 다
**`build_callbacks(args, ...)` 함수 안에서만** 쓰인다. 즉 `build_callbacks`를 원본
그대로 import해서 쓰되, `.timesteps`만 `T_ref`로 바꿔치기한 얕은 복사본을 넘기면
소스 수정 없이 T_ref/T_end 분리가 그대로 재현된다:

```python
# src/train_resume.py
import copy
from train import make_env_fn, build_callbacks  # 등, 재사용 가능한 모듈 최상위 함수
from eval import cache_path_for, split_bounds

schedule_args = copy.copy(args)
schedule_args.timesteps = T_REF          # build_callbacks 내부 3곳만 이 값을 봄
callbacks = build_callbacks(schedule_args, cache_paths, bounds, run_name, env_kwargs)
```

`ValidationCallback`/`CheckpointCallback`은 애초에 `args.timesteps`를 안 보므로
영향 없음. 실제 학습 스텝 수(`T_end`, delta 계산)는 `train_resume.py` 자신의 `args`로
별도 관리 — 서로 안 꼬임.

**재사용 가능 (import만, 수정 불필요)**: `make_env_fn`, `build_callbacks`,
`cache_path_for`, `split_bounds`, `MODE_SUBDIR`, `MODEL_DIR`, `LOG_DIR`, `LR_START`.

**복제해야 하는 글루 코드** (`main()` 안에 인라인이라 함수로 안 빠져 있어 그대로
못 가져옴, `train_resume.py`에 새로 작성):
- CPU affinity 고정 블록
- `DummyVecEnv`/`SubprocVecEnv` 구성 (env_fns 리스트 빌드 포함)
- `run_name` 타임스탬프 생성 + `configure_sb3_logger` 세팅
- 최종 모델 저장 블록
- 공유 하이퍼파라미터 argparse 플래그(`--symbols --workers --seed --stride --n-steps
  --batch-size --n-epochs --gamma --gae-lambda --clip-range --vf-coef --episode-days
  --eval-freq --kpi-smooth-window --eval-segments --checkpoint-freq --ent-coef-*
  --explore-bonus-* --leverage --cache-suffix --dummy-vec`)

**드리프트 리스크와 대응**: 두 파일이 같은 하이퍼파라미터 기본값을 각자 들고 있게
되므로, `train.py`의 기본값이 나중에 바뀌면(이 저장소는 leverage 1→3→5,
episode_days 30→14→30처럼 기본값이 실제로 자주 바뀐 이력이 있음) `train_resume.py`
쪽이 자동으로 안 따라온다. 이는 함정 4번(env kwarg 불일치가 obs shape은 그대로라
조용히 통과됨)과 같은 종류의 위험이 파일을 나누며 오히려 커진 것 — 그래서
**리섬 시 원 체크포인트와 반드시 일치해야 하는 값(`leverage`, `episode_days`,
`stride`, `n_steps`, `batch_size`, `workers`, `symbols`)**은 최초 설계에선 기본값
없는 필수 인자로 강제했었다. ⚠️ **2026-07-31 변경**: 사용자 요청으로
`--resume-from`/`--seed-best-info`만 필수로 남기고 나머지는 전부 0728-1924
베이스라인 기준 기본값을 부여하는 쪽으로 바뀜(§4 참고) — 매번 명시해야 하는
번거로움 대신, 다른 체크포인트를 resume할 땐 사용자가 그 런의 실제 설정으로
override할 책임을 진다. 아래 §5의 관련 함정 서술도 이 변경을 반영해 갱신했다.

### 3-1. 스케줄 기준 분리

두 개념 도입:
- `T_end` = `--timesteps`: 학습이 실제로 멈추는 절대 스텝 (예: 300M)
- `T_ref` = `--schedule-ref-total` (기본값 = `--timesteps`): ent_coef/explore_bonus의
  `*_frac` 인자가 곱해지는 기준 총스텝 (예: 기존 100M 유지)

fresh 런(resume 미지정)은 `T_ref = T_end`가 기본값이라 기존 동작과 완전히 동일.

### 3-2. lr piecewise 스케줄

```python
def make_lr_fn(lr_start, t_ref, t_end, knee_step=None, knee_value=None):
    if knee_step is None:
        return lambda pr: lr_start * pr          # 기존 단일 선형 (하위 호환)
    lr_knee = knee_value if knee_value is not None else lr_start * (1.0 - knee_step / t_ref)

    def lr_fn(pr):
        t = (1.0 - pr) * t_end
        if t < knee_step:
            return lr_start * (1.0 - t / t_ref)              # 무릎 전: 원 스케줄 그대로
        return lr_knee * max(0.0, (t_end - t) / (t_end - knee_step))
    return lr_fn
```

### 3-3. ent_coef / explore_bonus — 참조 상수만 교체

```python
# EntCoefSchedule._on_step
hold_steps = args.ent_coef_hold_frac * T_REF          # was: args.timesteps
frac = min((self.num_timesteps - hold_steps) / max(T_REF - hold_steps, 1), 1.0)

# build_callbacks
ExploreBonusSchedule(args.explore_bonus_start, args.explore_bonus_decay_frac * T_REF)
```

두 클래스 모두 이미 `min(..., 1.0)` 클램프가 있어 `t > T_ref` 구간에서 자동으로
종료값에 고정됨 — 로직 변경 없이 상수 교체만으로 충분.

### 3-4. `src/train_resume.py` 진입부

`train.py`의 `main()`을 분기하는 게 아니라, `train_resume.py`가 자체 `main()`을
갖고 항상 resume 경로만 수행한다 (fresh-run 분기 없음 — fresh는 여전히 `train.py`
담당).

```python
# src/train_resume.py
import copy
from train import make_env_fn, build_callbacks, MODE_SUBDIR, MODEL_DIR, LOG_DIR, LR_START
from eval import cache_path_for, split_bounds
from sb3_contrib import MaskablePPO

def main():
    args = parser.parse_args()   # --resume-from, --seed-best-info만 필수(2026-07-31 변경, §4).
                                  # --timesteps/--schedule-ref-total/--lr-knee-step 및
                                  # leverage/episode_days/stride/n_steps/batch_size/workers/
                                  # symbols는 전부 0728-1924 베이스라인 기준 기본값 보유

    # ... env_fns / vec_env 구성 (train.py main()에서 복제, §3-0 글루 코드 목록) ...

    lr_fn = make_lr_fn(LR_START, args.schedule_ref_total, args.timesteps,
                       args.lr_knee_step, args.lr_knee_value)
    model = MaskablePPO.load(
        args.resume_from, env=vec_env, device="cpu",
        custom_objects={"learning_rate": lr_fn, "lr_schedule": lr_fn},
        seed=args.seed, tensorboard_log=LOG_DIR,
    )
    start_step = model.num_timesteps
    delta = args.timesteps - start_step
    if delta <= 0:
        raise ValueError(f"--timesteps({args.timesteps:,}) <= 체크포인트 스텝({start_step:,})")

    schedule_args = copy.copy(args)
    schedule_args.timesteps = args.schedule_ref_total          # T_ref로 치환 (§3-0)
    callbacks = build_callbacks(schedule_args, cache_paths, bounds, run_name, env_kwargs)
    # ValidationCallback 상태 시드(§3-5)는 callbacks 안의 ValidationCallback 인스턴스에
    # best_score/kpi_history를 직접 주입 (build_callbacks 반환 리스트에서 꺼내 세팅)

    model.set_logger(...)
    model.learn(total_timesteps=delta, callback=callbacks, reset_num_timesteps=False)

    final_path = os.path.join(MODEL_DIR, MODE_SUBDIR, run_name, f"{run_name}_final")
    model.save(final_path)
```

### 3-5. ValidationCallback 상태 복원

- `_info.json`의 `btc_v9_kpi_z`(sel/mdd/msl 원시값, 예: 0728-1924 파일)는 저장돼 있으므로
  **현재 가중치(6:2:2)로 재계산**해 `best_score` 시드값으로 사용. 저장된 `btc_v9_kpi`
  원시값(1.6665)은 4:3:3 시절 값이라 그대로 쓰면 안 됨 (검산: `1.2569*.4+2.6227*.3+1.2564*.3
  =1.6665`, 6:2:2로는 `≈1.530`).
- `kpi_history`(deque)는 복구 불가 → 빈 상태 시작. resume 직후 첫 검증이 단일 샘플로
  smoothed 값을 만들어 스파이크 오판 문제가 재발할 수 있음 →
  **`len(kpi_history) == maxlen`일 때만 best 갱신 허용**하는 가드를 추가 (fresh 런
  초반의 동일 약점도 함께 방지됨).
- `--resume-reset-best` 플래그로 best를 `-inf`로 완전 리셋하는 옵션도 제공.
- 장기적으로는 best 저장 시 `{best_score, kpi_history, last_eval, num_timesteps}`를
  `run_state.json`에 함께 덤프해 resume 시 그대로 로드하는 방식이 정답. 위 항목들은
  과거(0728) 체크포인트를 위한 1회성 브릿지.

### 3-6. 저장 경로 분리 (2026-07-31 사용자 요청)

산출물을 `models|logs/v9_maskablerl/{run_name}`(fresh-run 트리)에 섞지 말고
`models|logs/resume/{원본 run_name}/{이번 resume run_name}`로 분리해달라는 요청.
`{원본 run_name}`은 `--resume-from` 경로의 파일명에서 `_best`/`_final`/`_<N>_steps`
접미사와 확장자를 제거해 뽑는다(`_orig_run_name()`) — 같은 원본에서 나온 여러
resume 시도가 한 폴더 아래 모인다.

문제는 `build_callbacks()`(train.py 원본, 무수정 import)가 **내부에서 직접**
`model_dir = os.path.join(MODEL_DIR, MODE_SUBDIR, run_name)`을 계산해 `_best.zip`/
체크포인트 저장 경로를 스스로 결정한다는 점 — `train_resume.py`가 만드는 `final`/
`runmeta.json` 경로와 달리 이 부분은 함수 인자로 주입할 수 있는 통로가 없다.
해결책: `train.py` 소스는 그대로 두고, **임포트한 모듈 객체의 전역 속성을 이 프로세스
안에서만 런타임에 패치**한다.

```python
_train_mod.MODEL_DIR = os.path.join(MODEL_DIR, "resume", orig_name)
_train_mod.MODE_SUBDIR = ""
```

Python 함수는 정의된 모듈의 `__globals__`에서 이름을 실행 시점에 조회하므로,
`train.build_callbacks()`가 나중에 실행될 때 위 패치된 값을 그대로 읽는다 — 디스크의
`train.py`는 한 글자도 바뀌지 않고, 이 프로세스가 끝나면 패치도 사라진다. `train.py`에서
`MODEL_DIR`/`MODE_SUBDIR`를 참조하는 곳은 `build_callbacks()` 하나뿐이라(다른 참조는
전부 우리가 호출하지 않는 `main()` 안에 있음) 부작용 없이 안전. `train_resume.py`
자신의 `model_dir`(final/runmeta 저장용)과 TensorBoard 로거 경로도 같은
`resume_model_root`/`resume_log_root` 변수로 통일해 계산한다.

스모크 검증(2026-07-31): 프레시 20k 체크포인트 → resume 24.5k 추가 실행 후
`models/resume/{원본}/{run_name}/{run_name}_final.zip`과
`logs/resume/{원본}/{run_name}/events.out.tfevents...`에 정확히 저장되고,
`models|logs/v9_maskablerl/`엔 이번 resume 산출물이 전혀 안 남는 것을 확인.

## 4. `src/train_resume.py` CLI 인자

**2026-07-31 변경(사용자 결정)**: §3-0에서는 원 체크포인트와 일치해야 하는 값들을
기본값 없는 필수 인자로 강제해 드리프트를 막으려 했으나, 실사용 요청에 따라
**`--resume-from`/`--seed-best-info` 둘만 필수로 남기고 나머지는 전부 기본값을
부여**하는 쪽으로 변경. 드리프트 방지라는 목적 자체보다 "매번 다 타이핑해야 하는"
번거로움이 실제로 더 크다는 판단. 대신 기본값을 임의로 잡지 않고 **`models/best/
v9_maskablerl_seed0_0728-1924_best.zip` 베이스라인의 실제 학습 설정**으로 고정했다
(leverage=3.0은 `quant_main`의 「RL 실전-백테스트 괴리 점검」 문서 + 커밋 `e83032f`의
`args.leverage is None and exit_mode=="rl" -> 3.0` 코드 해석을 교차 확인해 확정 —
당초 이 설계 문서에 5.0으로 잘못 적혀 있던 것을 정정함). **다른 체크포인트를
resume할 땐 이 기본값들이 안 맞을 수 있으므로 그 런의 실제 설정으로 반드시
override**해야 하고, 이는 여전히 사용자 책임 영역 — §3-0에서 우려했던 드리프트
위험 자체가 없어진 게 아니라 "필수 인자 강제"에서 "문서화된 기본값 + 사용자
주의"로 방어선을 옮긴 것.

| 인자 | 의미 | 기본값 |
|---|---|---|
| `--resume-from PATH` | 체크포인트 zip 경로 | 없음 — 필수 |
| `--seed-best-info PATH` | 원 체크포인트의 `_best_info.json` | 없음 — 필수 |
| `--timesteps N` | 최종 누적 절대 스텝(T_end) | 300,000,000 |
| `--schedule-ref-total N` | ent_coef/explore_bonus frac 기준 총스텝(T_ref) | 100,000,000 (원 런의 `--timesteps` 기본값) |
| `--lr-knee-step N` | lr 무릎 지점(절대 스텝) | 90,000,000 |
| `--lr-knee-value F` | 무릎 지점 lr. 미지정 시 `LR_START*(1-knee/ref)`로 자동 산출 | None(자동 산출) |
| `--resume-reset-best` | best_score를 `-inf`로 리셋 | False |
| `--leverage` | 원 체크포인트와 반드시 동일해야 함 | 3.0 |
| `--episode-days` `--stride` `--n-steps` `--batch-size` `--workers` `--symbols` | 원 체크포인트와 반드시 동일해야 함 | 30 / 1 / 2048 / 512 / 14 / [BTC,ETH] |
| 그 외(`--gamma --gae-lambda --clip-range --vf-coef --eval-freq --kpi-smooth-window --eval-segments --checkpoint-freq --ent-coef-* --explore-bonus-* --cache-suffix --dummy-vec`) | `train.py`와 동일 기본값 유지 (`--explore-bonus-start`만 예외: 0728-1924 런의 실제 rl 모드 해석값 0.005 — 어차피 96M 시점엔 이미 50M에서 감쇠 완료라 resume엔 무관) | `train.py`와 동일 |

### 사용 예 — 0728-1924 베이스라인 resume 최소 커맨드

```bash
python src/train_resume.py \
  --resume-from models/best/v9_maskablerl_seed0_0728-1924_best.zip \
  --seed-best-info models/best/v9_maskablerl_seed0_0728-1924_best_info.json
```

기본값들이 전부 이 베이스라인 기준이라 `--timesteps`/`--schedule-ref-total`/
`--lr-knee-step`/`--leverage` 등을 생략해도 위 §3-2 예시(100M→300M, 90M 무릎)와
동일하게 동작한다. 96M(체크포인트) 시점 lr은 2.914e-5 — "90M 무릎"은 과거 실제
궤적이 아니라 새 스케줄의 사후적 기준선이며, 실제로는 lr을 다시 올려
warm-restart하는 것과 같다. 재시작 직후 valid KPI가 일시적으로 흔들릴 수 있으나
500k 간격 검증으로 즉시 관측 가능하므로 별도 warmup 노브 없이 우선 시도 권장.

다른 체크포인트를 resume할 땐 아래처럼 전부 override:
```bash
python src/train_resume.py --resume-from PATH --seed-best-info PATH \
  --timesteps 300000000 --schedule-ref-total 100000000 --lr-knee-step 90000000 \
  --leverage 3.0 --episode-days 30 --stride 1 --n-steps 2048 --batch-size 512 \
  --workers 14 --symbols BTC-USDT-SWAP ETH-USDT-SWAP
```

## 5. 함정 목록 (구현 시 반드시 처리)

1. `custom_objects`에 `lr_schedule`만 넣으면 무효 — `learning_rate`를 오버라이드해야 함
   (§2, §3-4).
2. `reset_num_timesteps=False` + `total_timesteps` 그대로 넘기면 이중 가산 (§2, §3-4).
3. `--workers`가 원 런과 다르면 `load(env=...)`가 `n_envs`를 조용히 덮어써서
   (`data["n_envs"] = env.num_envs`) 에러 없이 통과하지만 롤아웃 크기(`n_steps × n_envs`)
   가 달라져 업데이트 semantics가 바뀜 → `--workers` 기본값(14)이 0728-1924 베이스라인과
   일치하니 그 체크포인트를 resume할 땐 안전하나, 다른 체크포인트는 반드시 override할 것
   (2026-07-31: 필수 인자 강제에서 문서화된 기본값으로 방어선 이동, §4 참고).
4. env 인자(leverage 등) 불일치가 obs shape은 그대로라 `check_for_correct_spaces`를
   통과해버림 → 런 시작 시 `{run_name}_runmeta.json`에 전체 args + git commit 저장은
   구현했으나(§3-4), 원 체크포인트의 runmeta 자체가 없어(0728-1924는 이 기능 이전 산출물)
   자동 하드-실패 검증은 아직 없음 — `--leverage`(기본 3.0) 등 기본값이 실제로 0728-1924와
   일치하도록 §4에서 교차 확인해뒀지만, 다른 체크포인트 resume 시엔 여전히 사용자가 직접
   확인/override해야 하는 수동 방어에 의존.
5. best_score 스케일 불일치 (4:3:3 → 6:2:2, §3-5).
6. resume도 새 `run_name`(새 타임스탬프)으로 분리 — TB에서 두 런이 num_timesteps
   기준으로 자연스럽게 이어져 보이고 원 런 산출물도 보존됨. 저장 경로는 **2026-07-31
   변경**으로 `models|logs/v9_maskablerl/{run_name}` 대신 `models|logs/resume/{원본
   run_name}/{이번 run_name}`로 분리(§3-6) — 원본 체크포인트별로 여러 resume 시도가
   한곳에 모이고, fresh-run 산출물 트리와도 섞이지 않는다.
7. `--final-split` 인자 설명(train.py:276~278)은 **수정하지 않는다** — `train.py` 자체를
   안 건드리기로 했으므로, "이어붙여도 사실상 안 배움"이라는 옛 서술은 `train.py`
   fresh-run 경로 단독 기준으로는 여전히 사실이라 남겨둬도 무방함. 실제 이어학습은
   `train_resume.py`로 별도 수행한다는 점만 이 설계 문서(§0)에 남긴다.
8. **파일 분리로 인한 기본값 드리프트** (§3-0) — `train.py`가 진화하며 하이퍼파라미터
   기본값이 바뀌어도 `train_resume.py`가 몰래 옛 값을 쓰게 될 위험. 최초엔 공유
   하이퍼파라미터를 필수 인자화해 침묵 실패를 막으려 했으나, 2026-07-31 사용자 요청으로
   `--resume-from`/`--seed-best-info`만 필수로 남기고 나머지는 0728-1924 베이스라인 기본값을
   부여하는 쪽으로 바뀌어 **이 드리프트 위험이 다시 열려 있음** — 다른 체크포인트를
   resume할 땐 leverage/episode_days/stride/n_steps/batch_size/workers/symbols를
   사용자가 반드시 수동으로 override해야 함(잊으면 조용히 잘못된 값으로 학습됨).
   `train_resume.py`를 오래 방치하면 그 자체가 유지보수 부담이 된다는 점도 감안할 것 —
   리섬을 자주 쓰게 되면 §3-5 장기 해법(`run_state.json`)과 함께 `train.py` 쪽에 정식으로
   통합하는 것도 재고할 만함.

## 6. 검증 절차

1. **기존 소스 무변경 확인**: `git diff src/train.py src/env.py src/eval.py
   src/algorithm.py src/prep_features.py`가 빈 결과여야 함 — 이 설계의 전제 조건.
   resume 관련 코드는 전부 `src/train_resume.py`에만 존재.
2. **resume 왕복**: `train.py`로 스모크 30k 학습 → `_final` 저장 →
   `python src/train_resume.py --resume-from ... --timesteps 60000
   --schedule-ref-total 30000 ...` 재실행. `num_timesteps`가 30000부터 이어지고
   `train/learning_rate`가 piecewise 식과 일치하는지 확인.
3. **결정론 대조**: `_best.zip`을 resume하면 `last_eval=0`이라 즉시 검증이 도는데,
   이 즉시 검증의 BTC 거래수/total_pnl/compound_mdd_pct가 `eval.py`로 같은 체크포인트를
   별도 평가한 값과 **정확히 일치**해야 함 (평가 경로는 deterministic argmax 전용이라
   완전 결정론적). v9_kpi 원시값으로 비교하지 말 것(6:2:2 변경으로 옛 info.json과
   원래 다름) — 거래 단위 지표로 대조.

## 7. 미결정 사항

- resume 소스: `_best.zip`(96.0M, valid 기준 최고, num_timesteps 자동 포함) 권장,
  `_final.zip`(100M) 대안.
- seed: 저장된 seed 그대로면 이어학습 구간이 원 런 초반과 동일한 에피소드 윈도
  순서를 재방문(무해하나 탈상관 없음). 데이터 재활용 관점에서 다른 seed를 줄지 결정 필요.
