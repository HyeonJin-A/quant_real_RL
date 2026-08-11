"""
V9 PPO 학습 (V9 Design.md 7장, 9장)

- 워커를 심볼별 배분 (예: 8워커 = BTC 4 + ETH 4). 2026-07-28부터 VecEnv 기본값은
  DummyVecEnv(단일 프로세스 순차 실행) — action masking 도입으로 스텝당 IPC 왕복이 2회가
  되면서 [64,64] 초소형 MLP에선 SubprocVecEnv 병렬화 이득이 역전됨(fps 약 9,900→16,800).
  --no-dummy-vec으로 기존 SubprocVecEnv 병렬 경로 사용 가능.
- 학습 데이터: 시계열 70% train 구간, 랜덤 시작 30일 에피소드
  (2026-07-19 60일로 확대했다가 2026-07-20 30일 복귀 — 60일 런(0719 오전)이 칼손절
  스캘핑 분지로 붕괴(승률 16%, PF 0.86)했고, 에피소드만 30일로 되돌린 0719-1459 런이
  스윙형/승률 27%로 정상 복귀한 실측 근거. 업데이트당 레짐 다양성 감소가 유력 원인)
- 검증 콜백: eval_freq마다 검증셋(15%) 전체 결정론적 롤아웃 →
  거래수/승률/PnL/복리 지표를 TensorBoard 기록,
  BTC 월별 복리 log-multiple의 평균−표준편차가 최고인 체크포인트를 best로 저장
  (2026-07-20, 거래수 가드 포함 — ValidationCallback 주석 참고)
- lr 3e-4 → 0 선형 감쇠

rl 모드(보유 중 풀 컨트롤, MaskablePPO) 단일 모드. 붕괴 이력(2026-07-16 seed0, 검증 콜백
최대 4회 연속 거래 0건 — "진입=확실한 수수료 비용, Hold=항상 0"이라는 구조적 비대칭)에
대응해 2026-07-20 재설계: 방향은 fade 고정 + explore_bonus(Enter 시 임시 보너스, 학습
커리큘럼 전용, --explore-bonus-start에서 --explore-bonus-decay-frac 지점까지 선형으로
0에 수렴, cum_pnl/평가 점수엔 전혀 반영 안 됨) 도입.

2026-07-30: 레거시 exit_mode="rule"/"adaptive" 및 그 전용 장치(LogStdClampCallback,
LeverageMaxSchedule 등) 완전 제거 — "rl"이 유일한 지원 모드.

사용법:
  python src/rl_v9/train_v9.py --seed 0                          # 기본 100M 스텝
  python src/rl_v9/train_v9.py --timesteps 30000 --workers 2 --dummy-vec --cache-suffix _recent120d  # 스모크
"""
import os
import sys
import json
import argparse
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

# 2026-07-24: torch 기본 intra-op 스레드 수(보통 4, nproc 기반 자동설정)가 [64,64] 초소형
# MLP엔 오히려 손해 — 스레드 동기화 오버헤드가 실제 연산량을 넘어서 메인 프로세스가 항상
# ~400%(코어 4개)를 잡아먹으면서도 SubprocVecEnv 워커들은 대부분 유휴 상태였음(실측: CPU
# 사용률 260%대에서 병목 진단). 1로 낮추자 fps가 스모크 테스트 기준 약 2450→3200(+30%) 개선.
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(__file__))
from env import TradingEnvV9, FEATURE_NAMES  # noqa: E402
from eval import (cache_path_for, split_bounds, run_policy_on_ranges, compute_metrics,  # noqa: E402
                  compound_metrics, monthly_sel_score, v10_kpi, MIN_TRADES_PER_MONTH,
                  MAX_COMPOUND_MDD_PCT, MIN_WORST_EQUITY, CACHE_VER)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

LR_START = 3e-4


MODE_SUBDIR = "v10_maskablerl"  # 2026-08-03: RSI-다이버전스 버그 2종 수정(algorithm.py/prep_features.py,
                                # RSI-div-개편이전-핵심결함.md) 이후 첫 학습 라인 — 이전 v9_maskablerl/
                                # 산출물과 피처 캐시가 달라 비교 기준이 바뀌므로 이름으로 세대를 구분.
                                # 정책 클래스(MaskablePPO)·행동공간은 불변, 구분은 순전히 명명 목적.
                                # (구 v9_maskablerl/ 산출물은 과거 기록으로 그대로 보존)


def make_env_fn(cache_path, lo, hi, episode_len_rows, decision_stride, seed, env_kwargs, worker_idx=0,
                pin_core=True):
    def _init():
        # 2026-07-25: 워커 프로세스를 코어 1번부터 순차 고정 (부모 Core 0 침범 방지)
        # 전체 CPU 코어 수(num_cpus)를 반영하여 Core 1 ~ (num_cpus-1) 범위에서 순환 배정
        # 2026-07-28: pin_core=False(DummyVecEnv)면 건너뜀 — 별도 프로세스가 없어 이 호출이
        # 메인 프로세스의 어피니티를 워커 수만큼 덮어써버리고(마지막 워커 값이 남음) 결과적으로
        # Core 0 고정이 풀린다. DummyVecEnv는 전 env가 메인 프로세스에서 순차 실행되므로
        # 메인의 Core 0 고정(아래 main() 참고)만으로 충분하다.
        if pin_core and hasattr(os, "sched_setaffinity"):
            try:
                num_cpus = os.cpu_count() or 1
                target_core = 1 + (worker_idx % (num_cpus - 1)) if num_cpus > 1 else 0
                os.sched_setaffinity(0, {target_core})
            except Exception:
                pass

        from stable_baselines3.common.monitor import Monitor
        env = TradingEnvV9(
            cache_path, start_idx=lo, end_idx=hi,
            episode_len_rows=episode_len_rows, decision_stride=decision_stride,
            **env_kwargs,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def build_callbacks(args, cache_paths, bounds, run_name, env_kwargs):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    model_dir = os.path.join(MODEL_DIR, MODE_SUBDIR, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # 2026-08-09: 피처 후진제거(ablation) 메타데이터 사이드카 — exclude_features가 빈 경우도
    # 포함해 모든 런에 저장한다. eval.py --model이 이 파일을 자동으로 읽어 학습 때와 동일한
    # exclude_features로 env를 재구성한다 (없으면 구 체크포인트로 간주해 전체 18피처 기본값).
    # 목적: 서로 다른 두 exclude_features가 우연히 같은 관측 차원이 되는 경우에도(예: 각각
    # 하나씩 제외한 두 실험) 이름 목록으로 정확히 구분해 "모양은 맞지만 의미가 다른" 조용한
    # 오염을 방지 (2026-08-03 캐시 버전 사고와 동일한 실패 유형).
    exclude_features = sorted(env_kwargs.get("exclude_features") or [])
    with open(os.path.join(model_dir, f"{run_name}_features.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exclude_features": exclude_features,
            "obs_dim": len(FEATURE_NAMES) - len(exclude_features),
            "cache_ver": getattr(args, "cache_ver", None) or CACHE_VER,
            "leverage": args.leverage,
        }, f, indent=2)

    class EntCoefSchedule(BaseCallback):
        """--ent-coef-hold-frac 지점까지 시작값 고정 유지 후, 나머지 구간에서만 선형 감쇠.
        (기존 처음부터 선형 감쇠 방식이 두 번의 붕괴 모두에서 무효였던 것에 대한 대응, 2026-07-16)"""
        def _on_step(self):
            hold_steps = args.ent_coef_hold_frac * args.timesteps
            if self.num_timesteps <= hold_steps:
                self.model.ent_coef = args.ent_coef_start
            else:
                frac = min((self.num_timesteps - hold_steps) / max(args.timesteps - hold_steps, 1), 1.0)
                self.model.ent_coef = args.ent_coef_start + (args.ent_coef_end - args.ent_coef_start) * frac
            return True

    class ExploreBonusSchedule(BaseCallback):
        """Enter 시 실현손익과 별개로 붙는 임시 탐험 보너스.
        decay_frac 지점까지 선형으로 0에 수렴 — V8과 무관하게 콜드스타트만 겨냥한 장치.
        VecEnv.env_method("set_curriculum", ...)로 모든 워커의 env.explore_bonus를 갱신
        (롤아웃 시작 시 1회, 저비용). ⚠️ set_attr이 아니라 반드시 env_method를 써야 함 —
        set_attr(name, value)는 Monitor 등 Wrapper 표면에 setattr(env_i, name, value)를
        호출해 그림자 속성만 만들고 내부 TradingEnvV9까지 안 닿는 SB3 자체 버그가 있음
        (get_attr은 get_wrapper_attr로 래퍼를 뚫고 들어가 값이 바뀐 것처럼 보이는 착시를
        주지만, 실제 step()이 읽는 값은 그대로였음 — 2026-07-17, 두 설정을 완전히 다르게
        줘도 결과가 소수점까지 동일하게 나오는 것으로 발견/확인)."""
        def __init__(self, bonus_start, decay_timesteps):
            super().__init__()
            self.bonus_start = bonus_start
            self.decay_timesteps = max(decay_timesteps, 1)

        def _on_rollout_start(self):
            frac = min(self.num_timesteps / self.decay_timesteps, 1.0)
            bonus = self.bonus_start * (1.0 - frac)
            self.training_env.env_method("set_curriculum", explore_bonus=float(bonus))

        def _on_step(self):
            return True

    class ValidationCallback(BaseCallback):
        # 2026-07-28 버그 수정: verbose 기본값 1. SB3의 BaseCallback.init_callback()은
        # model.verbose를 콜백에 전파하지 않아(self.model만 세팅) self.verbose가 0으로 남았고,
        # 그 결과 아래 `if self.verbose:` 블록 두 개(검증 요약 출력, "new best" 저장 알림)가
        # 학습 내내 한 번도 실행되지 않는 죽은 코드였음 — best 체크포인트가 언제 갱신됐는지
        # (혹은 하드게이트에 막혀 한 번도 저장 안 됐는지) 로그만 봐서는 알 수 없었다.
        def __init__(self, eval_freq, kpi_smooth_window=3, verbose=1):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.last_eval = 0
            self.best_score = -np.inf
            # 2026-07-29: best 선택에 쓰는 v9_kpi를 원시값 대신 최근 N회 이동평균으로 교체 —
            # 단일 체크포인트의 우연한 스파이크가 실제로는 더 꾸준했던 인접 구간을 근소 차이로
            # 이겨버리는 문제 실측 확인(0729-1938 런: 73M 스파이크 1.367이 95.5~96.0M의
            # 연속 우수 구간(1.1대, sel/mdd 둘 다 73M보다 좋음)을 제치고 best로 선택됨).
            self.kpi_history = deque(maxlen=kpi_smooth_window)

        def _on_step(self):
            if self.num_timesteps - self.last_eval < self.eval_freq:
                return True
            self.last_eval = self.num_timesteps
            btc_sel = None  # BTC 월별 log-multiple (참고 기록용, 선택 기준 아님)
            btc_logs_m = None
            btc_kpi = None  # BTC v10_kpi — best 선택 점수 (2026-08-06부터 선택 기준)
            btc_kpi_detail = None
            btc_mdd_pct = None   # ⚠️ 루프 밖에서 m/cm을 쓰면 마지막 심볼(ETH) 값이므로 별도 보관
            btc_msl = None
            btc_eligible = False
            # 2026-07-28: 심볼별 순차 롤아웃 → 독립 레인 배치 1회 호출로 교체
            # (`검증 롤아웃 속도 최적화.md`) — 결과는 거래 목록까지 완전 동일, 속도만 개선.
            targets = []
            for sym, path in cache_paths.items():
                lo, hi = bounds[path]["valid"]
                if hi - lo < 100:
                    continue
                targets.append((sym, path, lo, hi))
            trades_by_sym = run_policy_on_ranges(
                self.model, targets,
                n_segments=args.eval_segments, decision_stride=args.stride,
                **env_kwargs,
            )
            for sym, path, lo, hi in targets:
                trades = trades_by_sym[sym]
                ts = np.load(path)["ts_1m"]
                m = compute_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
                for key in ("trades", "trades_per_month", "win_rate", "total_pnl",
                            "max_single_loss", "pnl_std",
                            "near_liq_n", "near_liq_pct", "max_upnl", "worst_equity"):
                    self.logger.record(f"valid/{sym}/{key}", m[key])
                # 복리(전액 재투입) 지표 — 실전 운용 방식 기준 참고용 기록 (2026-07-19 추가).
                # 모델 선택에는 compound_mdd_pct만 관여(v10_kpi 점수엔 안 섞이고 하드게이트 전용).
                cm = compound_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
                self.logger.record(f"valid/{sym}/compound_multiple", cm["multiple"])
                self.logger.record(f"valid/{sym}/compound_mdd_pct", cm["mdd_pct"])
                # 2026-07-28 추가. compound_multiple은 "거래당 엣지 × 거래 횟수"가 뒤섞여 있어
                # 단독으로는 오해를 부른다(실측: 거래당 엣지가 더 나쁜 쪽이 거래를 61% 더 해서
                # multiple은 18.6배 높게 표시됨). 아래 둘로 그 두 축을 분리해서 본다:
                #   monthly_growth      — 달력 시간으로 정규화 (거래 빈도 효과를 정당하게 포함)
                #   geo_mean_per_trade  — 거래 횟수로 정규화 (순수 엣지 품질, 빈도 효과 제거)
                self.logger.record(f"valid/{sym}/monthly_growth", cm["monthly_growth"])
                self.logger.record(f"valid/{sym}/geo_mean_per_trade", cm["geo_mean_per_trade"])
                if sym == "BTC-USDT-SWAP":
                    # 모델 선택 기준: BTC 월별 복리 log-multiple 평균−표준편차 (2026-07-20).
                    # 근거(0719-1459 런 실측): 구 v9_score(승률×1000 항 지배)는 total_pnl +28
                    # (사실상 본전, top1 제거 시 적자)인 9.5M이 +418인 50M을 제치고 best로
                    # 뽑혔고, 실전 운용 방식(전액 재투입 복리)과 선택 목적이 어긋나 복리 MDD
                    # 99% 체크포인트가 걸러지지 않았음. 월 12표본이라 분기 4표본보다 분산
                    # 추정도 안정적. 레짐 편중 감점(−std)이라는 취지는 그대로 계승.
                    # (2026-07-25: v9_score 자체를 TB 기록에서도 완전 폐기)
                    # 2026-08-06: sel_monthly_log는 더 이상 선택 기준이 아니라 참고 기록용
                    # (v10_kpi가 대체) — 그래도 월별 복리 궤적을 TB/best_info에서 보기 위해 유지.
                    btc_sel, btc_logs_m = monthly_sel_score(trades, int(ts[lo]), int(ts[hi - 1]))
                    # 선택 점수: v9_kpi(수익성:낙폭:꼬리위험 6:2:2 가중합) → v10_kpi(거래단위
                    # t-통계량)로 교체 (eval.py v10_kpi 주석에 배경 상세). 저거래 체크포인트가
                    # 표본분산이 우연히 작아 최고점을 찍는 왜곡이 반복 관측돼, 표본 크기가
                    # 점수식에 직접 반영되는 방식으로 완전히 새로 설계.
                    btc_mdd_pct, btc_msl = cm["mdd_pct"], m["max_single_loss"]
                    btc_kpi, btc_kpi_detail = v10_kpi(trades, detail=True)
                    # "무거래 = 월배수 1.0 = 중립"이 손실 정책보다 우대되는 함정 차단:
                    # 월평균 거래수 미달이면 best 후보 자격 자체를 박탈 (합격 기준 ①과 동일 문턱)
                    # 근접청산 건수(near_liq_n)가 0이 아니면 자격 박탈 (2026-07-26 추가):
                    # 점수만으로는 근접청산 위험을 감점만 할 뿐 걸러내지 못해, 월별 변동성이
                    # 우연히 낮게 나온 위험한 체크포인트가 best로 뽑힐 수 있음.
                    # 복리 MDD가 MAX_COMPOUND_MDD_PCT(70%) 이상이면 자격 박탈 (2026-07-28 추가):
                    # v10_kpi는 낙폭을 전혀 안 보므로(거래단위 t-통계량뿐) 이 하드게이트가 낙폭
                    # 통제의 유일한 수단이다 — v9_kpi 때보다 오히려 이 게이트의 역할이 커짐.
                    # ⚠️ 이 게이트를 한 번도 통과하지 못한 런은 _best.zip이 아예 생성되지 않는다
                    # (의도된 동작 — 그런 런은 애초에 실전 부적합. 실측: leverage=10 런 0727-0934는
                    # MDD 최저 88.3%로 전멸).
                    # worst_equity(월별 최악의 월말 잔고) < 60이면 자격 박탈 (2026-08-06 추가):
                    # compound_mdd_pct는 전체 구간 통짜 복리 낙폭이라 "가장 나쁜 한 달"만 따로
                    # 보진 못함 — 전체 MDD가 낮아도 특정 달에 계좌가 반토막 나는 체크포인트를
                    # v10_kpi(거래단위 t-통계량)만으로는 못 걸러내므로 별도 게이트로 보강.
                    btc_eligible = (m["trades_per_month"] >= MIN_TRADES_PER_MONTH
                                     and m["near_liq_n"] == 0
                                     and btc_mdd_pct < MAX_COMPOUND_MDD_PCT
                                     and m["worst_equity"] >= MIN_WORST_EQUITY)
                if self.verbose:
                    print(f"[valid @{self.num_timesteps:,}] {sym}: "
                          f"trades={m['trades']} pnl={m['total_pnl']:+.1f}")
            # 모델 선택은 BTC 단독 (2026-07-19: min(BTC,ETH) 기준은 만성적으로 ETH에 끌려
            # 내려가 BTC에서 잘하는 체크포인트를 놓쳤음. ETH는 지표 기록만 유지하는 참고용).
            if btc_sel is not None:
                self.logger.record("valid/BTC-USDT-SWAP/sel_monthly_log", btc_sel)
                self.logger.record("valid/BTC-USDT-SWAP/v10_kpi", btc_kpi)
                # 항별 상세(t-통계량 분해: mean/std/n) — TB에서 바로 보기 위함
                for k, v in btc_kpi_detail.items():
                    self.logger.record(f"valid/BTC-USDT-SWAP/v10_kpi_{k}", v)
                # 2026-07-29: best 비교는 원시 v10_kpi가 아니라 최근 N회(kpi_history) 이동평균으로
                # 수행 — 고립된 스파이크 1회보다 꾸준히 높았던 구간을 우대. 자격 게이트(거래수/
                # 근접청산/MDD)는 여전히 "현재 시점" 기준으로 적용(과거 실격 구간이 지금 좋아졌다고
                # 그 흔적만으로 통과시키지 않음).
                self.kpi_history.append(btc_kpi)
                smoothed_kpi = float(np.mean(self.kpi_history))
                self.logger.record("valid/BTC-USDT-SWAP/v10_kpi_smoothed", smoothed_kpi)
                if btc_eligible and smoothed_kpi > self.best_score:
                    self.best_score = smoothed_kpi
                    path = os.path.join(model_dir, f"{run_name}_best")
                    self.model.save(path)
                    with open(path + "_info.json", "w", encoding="utf-8") as f:
                        json.dump({"timesteps": self.num_timesteps,
                                   "btc_v10_kpi": round(btc_kpi, 4),
                                   "btc_v10_kpi_smoothed": round(smoothed_kpi, 4),
                                   "btc_v10_kpi_detail": {k: round(v, 4) for k, v in btc_kpi_detail.items()},
                                   "btc_sel_monthly_log": round(btc_sel, 4),
                                   "btc_compound_mdd_pct": round(btc_mdd_pct, 2),
                                   "btc_max_single_loss": round(btc_msl, 2),
                                   "btc_monthly_multiples": [round(float(np.exp(l)), 3) for l in btc_logs_m]}, f)
                    if self.verbose:
                        print(f"[valid] new best (BTC v10_kpi {btc_kpi:+.4f}, smoothed {smoothed_kpi:+.4f} | "
                              f"n={btc_kpi_detail['n']} mdd {btc_mdd_pct:.1f}% msl {btc_msl:+.1f}) -> {path}.zip")
            return True

    callbacks = [EntCoefSchedule(), ValidationCallback(args.eval_freq, kpi_smooth_window=args.kpi_smooth_window)]
    if args.explore_bonus_start > 0:
        callbacks.append(ExploreBonusSchedule(
            args.explore_bonus_start, args.explore_bonus_decay_frac * args.timesteps,
        ))
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.workers, 1), 1),
            save_path=model_dir, name_prefix=run_name,
        ))
    return callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    parser.add_argument("--final-split", action="store_true",
                        help="배포 전 최종학습용 95/5(train/valid) 분할 사용 (기본은 개발용 70/15/15). "
                             "이어학습이 아니라 처음부터 새로 학습해야 함 — 기존 체크포인트는 lr/ent_coef/"
                             "explore_bonus 스케줄이 이미 소진돼 있어 이어붙여도 사실상 안 배움 (2026-07-23)")
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--main-core", type=int, default=0, help="메인(부모) 프로세스를 고정할 CPU 코어 번호")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO 롤아웃 버퍼 크기")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO 미니배치 크기")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO 최적화 에포크 수")
    parser.add_argument("--gamma", type=float, default=0.999, help="할인 계수")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE 람다")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO 클리핑 범위")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="가치 함수 손실 계수")
    parser.add_argument("--episode-days", type=int, default=30,
                        help="학습 에피소드 길이(일). 2026-07-20: 60→30 복귀 (60일 런 붕괴 실측, 모듈 독스트링 참고). "
                             "2026-07-21: rl 모드 기본값을 14일로 잠깐 내렸다가(리셋 빈도 확보 목적) 다음 런 성과가 "
                             "더 나빠 30일로 원복 — 다만 그 런엔 explore_bonus(0.15)도 그대로 남아있어 원인이 "
                             "완전히 격리되진 않음 (6장 이력 참고)")
    parser.add_argument("--eval-freq", type=int, default=500_000)
    parser.add_argument("--kpi-smooth-window", type=int, default=3,
                        help="best 체크포인트 선택에 쓰는 v10_kpi 이동평균 윈도(검증 회차 수, 기본 3). "
                             "단일 검증 스냅샷의 우연한 스파이크가 근소한 차이로 연속 우수 구간을 이기는 "
                             "문제 대응 (2026-07-29) — 최근 N회 v10_kpi의 평균이 best_score보다 높을 때만 갱신")
    parser.add_argument("--eval-segments", type=int, default=1)  # 2026-07-19: 세그먼트 분할 폐기 (경계 오차)
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--ent-coef-start", type=float, default=0.01,
                        help="탐험 강도 초기값 (기본 0.01)")
    parser.add_argument("--ent-coef-end", type=float, default=0.005,
                        help="탐험 강도 종료값 (기본 0.005)")
    parser.add_argument("--ent-coef-hold-frac", type=float, default=0.7,
                        help="전체 스텝 중 이 비율 지점까지 ent_coef_start를 그대로 유지 후 감쇠 시작 (기본 0.7. "
                             "2026-07-21: 0.85로 상향 시도했으나 같은 런에서 explore_bonus(0.15, 50%%까지 유지)가 "
                             "과매매 붕괴를 오히려 25M까지 더 길게 끌고 간 정황이 드러나 원인이 ent_coef가 아닐 "
                             "가능성이 높아져 0.7로 원복 — explore_bonus 쪽을 먼저 조정해보기로 함)")
    parser.add_argument("--explore-bonus-start", type=float, default=0.003,
                        help="Enter 시 붙는 임시 탐험 보너스 초기값 (0이면 비활성화). 06-16 붕괴(진입=확실한 비용, "
                             "Hold=항상 0의 구조적 비대칭)에 대한 대응. "
                             "2026-07-21: 레버리지 낮음(기본 1)에서는 0.15가 거꾸로 진입 시 확정 수수료비용"
                             "(leverage×fee_rate, 레버리지1 기준 0.0005)의 300배에 달해 실제 손익 신호를 "
                             "덮어버림 — '진입을 자주'만 배우고 '잘'은 못 배우게 만듦. 0.001(수수료의 약 2배)로 "
                             "재하향한 런(0722-0702)이 이 세션 최초로 valid+test 동시 통과. 2026-07-24: 기본 "
                             "레버리지가 3으로 오르며 확정 수수료비용도 3배(0.0015)가 돼, 같은 2배 비율 유지를 "
                             "위해 0.003으로 재조정. 2026-07-28: 기본 레버리지 3→5에 맞춰 같은 비율로 0.005로 "
                             "재조정 (6장 이력 참고)")
    parser.add_argument("--explore-bonus-decay-frac", type=float, default=0.5,
                        help="전체 스텝 중 이 비율 지점에서 탐험 보너스가 0으로 수렴 (기본 0.5)")
    parser.add_argument("--leverage", type=float, default=3.0,
                        help="고정 레버리지 (기본 5.0, 2026-07-28 — 3.0은 2026-07-24 0724-0001 베이스라인 "
                             "값이었고, leverage=1 대비 test에서도 total_pnl/PF/복리 대폭 개선 확인. 10배는 "
                             "청산 위험이 과도해 실전 배제로 결론나 5.0 채택, 5장·6장 참고). eval.py 평가 시 "
                             "학습 때와 반드시 동일 값을 지정해야 함 — liq_dist 관측 피처가 leverage에 의존.")
    parser.add_argument("--cache-suffix", default="", help="스모크용 캐시 suffix (예: _recent120d)")
    parser.add_argument("--cache-ver", default=None,
                         help="캐시 버전 (예: v9d2). 미지정 시 eval.CACHE_VER(최신) 사용 — "
                              "2026-08-04: train.py에 이 옵션이 없어 eval.py의 --cache-ver와 "
                              "달리 항상 eval.CACHE_VER 하드코딩 값만 쓰던 누락을 뒤늦게 발견해 추가")
    parser.add_argument("--dummy-vec", action=argparse.BooleanOptionalAction, default=True,
                        help="DummyVecEnv(단일 프로세스) 사용 — 2026-07-28부터 기본값. action masking 도입으로 "
                             "MaskablePPO가 매 스텝 env.step()과 별도로 action_masks()를 한 번 더 조회하면서 "
                             "SubprocVecEnv의 IPC 왕복이 스텝당 2회가 됐고, [64,64] 초소형 MLP라 연산량보다 "
                             "통신 비용이 커서 병렬화 이득이 역전됨(실측 fps 약 9,900 → 16,800, 약 1.7배). "
                             "--no-dummy-vec으로 기존 SubprocVecEnv 병렬 경로 사용 가능")
    parser.add_argument("--exclude-features", nargs="*", default=[],
                        help="관측에서 제외할 피처 이름(공백 구분, env.FEATURE_NAMES 중). "
                             "기본 빈 리스트=전체 18개 사용 (2026-08-09 피처 후진제거 실험용)")
    args = parser.parse_args()

    bad_features = sorted(set(args.exclude_features) - set(FEATURE_NAMES))
    if bad_features:
        raise SystemExit(f"--exclude-features: 알 수 없는 이름 {bad_features}; 허용: {list(FEATURE_NAMES)}")

    env_kwargs = {"leverage": args.leverage, "exclude_features": tuple(sorted(set(args.exclude_features)))}

    # 2026-07-25: 메인(부모) 프로세스를 특정 코어에 100% 고정 (코어 널뛰기 완벽 방지). 2026-07-29: 상수(1) -> --main-core로 파라미터화.
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {args.main_core})
            num_cpus = os.cpu_count() or 1
            if args.dummy_vec:
                print(f"[CPU Affinity] Main process -> Core {args.main_core} | DummyVecEnv: {args.workers} envs 전부 메인에서 순차 실행 (워커 핀 없음)")
            else:
                print(f"[CPU Affinity] Main process -> Core {args.main_core} | {args.workers} workers -> Cores 1~{min(args.workers, num_cpus - 1) if num_cpus > 1 else 0}")
        except Exception as e:
            print(f"[CPU Affinity] Failed to set affinity: {e}")

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    # 2026-07-27: action masking (`V9 Design TODO - Action Masking.md`) —
    # 무포지션+Close, 보유중+Enter처럼 상태에 안 맞는 행동을 후보에서 물리적으로 제거.
    # env.py의 TradingEnvV9.action_masks()를 Monitor 래퍼가 그대로 위임하므로 별도
    # ActionMasker 래퍼 불필요(sb3_contrib get_action_masks가 env_method로 직접 호출).

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    cache_paths = {s: cache_path_for(s, args.cache_suffix, cache_ver=args.cache_ver) for s in args.symbols}
    for s, p in cache_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"cache not found for {s}: {p} — run prep_features_v9.py first")
    bounds = split_bounds(list(cache_paths.values()), final=args.final_split)

    episode_len_rows = args.episode_days * 1440 // args.stride * args.stride
    env_fns = []
    symbols = list(cache_paths.keys())
    for w in range(args.workers):
        sym = symbols[w % len(symbols)]  # 워커를 심볼별 균등 배분
        path = cache_paths[sym]
        lo, hi = bounds[path]["train"]
        env_fns.append(make_env_fn(path, lo, hi, episode_len_rows, args.stride, args.seed * 1000 + w, env_kwargs,
                                   worker_idx=w, pin_core=not args.dummy_vec))

    vec_env = DummyVecEnv(env_fns) if args.dummy_vec else SubprocVecEnv(env_fns, start_method="spawn")

    # 2026-07-19: 런 이름에 시작 시각을 붙여 유니크화 — 재시작 시 TB 런 디렉토리와
    # models/ 체크포인트(best/final 포함)가 이전 런 산출물을 덮어쓰는 사고 방지.
    # 2026-07-20: 서버 로컬시간(UTC) 대신 KST로 표기.
    log_subdir = MODE_SUBDIR
    run_name = f"v10_maskablerl_seed{args.seed}_{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%m%d-%H%M')}"
    print(f"run_name: {run_name}")
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lambda progress_remaining: LR_START * progress_remaining,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,               # 1m 스텝 기준 유효 horizon ~16시간 (V9 Design.md 6장)
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef_start,   # 콜백에서 --ent-coef-end로 선형 감쇠
        vf_coef=args.vf_coef,
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        device="cpu",              # CPU 강제 사용 (GPU 미활용 설정)
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # 2026-07-20: model.learn(tb_log_name=...)에 맡기면 SB3가 항상 "{run_name}_{N}"으로
    # run-id를 붙임(런 이름이 이미 타임스탬프로 유니크라 불필요한 접미사). model.learn() 전에
    # 커스텀 로거를 직접 설정해두면 SB3가 이걸 존중하고 configure_logger()를 재호출하지
    # 않아(_custom_logger 플래그, SB3 소스 확인) 정확히 "{run_name}" 경로로 남는다.
    from stable_baselines3.common.logger import configure as configure_sb3_logger
    model.set_logger(configure_sb3_logger(os.path.join(LOG_DIR, log_subdir, run_name), ["stdout", "tensorboard"]))

    callbacks = build_callbacks(args, cache_paths, bounds, run_name, env_kwargs)
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_path = os.path.join(MODEL_DIR, log_subdir, run_name, f"{run_name}_final")
    model.save(final_path)
    print(f"final model saved: {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
