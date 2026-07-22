"""
V9 PPO 학습 (V9 Design.md 7장, 9장)

- SubprocVecEnv 워커를 심볼별 배분 (예: 8워커 = BTC 4 + ETH 4)
- 학습 데이터: 시계열 70% train 구간, 랜덤 시작 30일 에피소드
  (2026-07-19 60일로 확대했다가 2026-07-20 30일 복귀 — 60일 런(0719 오전)이 칼손절
  스캘핑 분지로 붕괴(승률 16%, PF 0.86)했고, 에피소드만 30일로 되돌린 0719-1459 런이
  스윙형/승률 27%로 정상 복귀한 실측 근거. 업데이트당 레짐 다양성 감소가 유력 원인)
- 검증 콜백: eval_freq마다 검증셋(15%) 전체 결정론적 롤아웃 →
  거래수/승률/PnL/복리/v9_score를 TensorBoard 기록,
  BTC 월별 복리 log-multiple의 평균−표준편차가 최고인 체크포인트를 best로 저장
  (2026-07-20, 거래수 가드 포함 — ValidationCallback 주석 참고)
- lr 3e-4 → 0 선형 감쇠

exit_mode 기본값 "adaptive" (V9 Design.md 9장 Fallback B 확장, "진입만 학습" + 진입 시
레버리지/손절/반익절/완익절까지 정책이 직접 결정):
  붕괴 이력 3건 —
  ① 2026-07-16 seed0 exit_mode="rl"(풀 컨트롤): 검증 콜백 최대 4회 연속 거래 0건.
     ent_coef 0.01→0.001을 0.03→0.01로 상향, 에피소드 길이를 30일→1일로 롤아웃 배치
     이하로 단축, 둘 다 시도했지만 무효. 원인: 진입에 즉시 수수료 비용이 붙는 반면
     Hold는 항상 0이라, 무작위 초기화 시점부터 Hold가 구조적으로 살짝 유리해서 학습이
     진행될수록 그 우위가 스스로 강화되는 함정.
  ② 2026-07-16 seed0 exit_mode="adaptive"(5차원 연속 액션, ent_coef 0.03→0.01 처음부터
     선형 감쇠): 100만~150만 스텝 구간엔 min V8 Score -626→-14 개선, 거래도 정상 발생.
     그러나 이후 250만~350만 스텝 4회 연속 거래 0건. V8 사전학습(BC) 워밍업도 검토했으나
     V8 자체가 거래 희소/편차 문제로 폐기 대상이었으므로 그 근처에서 시작하면 같은 함정을
     다른 모양으로 재현할 위험이 있어 기각.
  ③ (②의 재시도) ent_coef를 --ent-coef-hold-frac(0.7) 지점까지 고정 유지 + explore_bonus
     (Enter 시 임시 보너스, 학습 커리큘럼 전용) 추가: 초반 100만 스텝은 최고 실적(BTC
     +165/ETH +708)이었으나 600만 스텝까지 지켜본 결과 거래수가 0~200 사이를 계속
     오르내리는 불안정 패턴이었고, entropy_loss는 -7→-62까지 단 한 번도 안 꺾이고 계속
     "더 음수"로 감. **여기서 SB3 소스(entropy_loss = -mean(entropy))로 부호를 재확인한
     결과 오진단이었음이 드러남**: entropy_loss가 더 음수 = entropy가 더 큼(이산 액션은
     엔트로피 상한이 ln(n)이라 반대로 해석해도 우연히 맞았지만, 연속 Gaussian은 엔트로피에
     상한이 없어 log_std가 무한정 커질 수 있음). 즉 "확신을 갖고 굳어버림"이 아니라
     **표준편차가 통제불능으로 폭주**해 액션이 -1/+1 극단(레버리지 항상 1 또는 100 등)으로
     계속 튀며 학습 신호 자체가 오염된 것 — ent_coef=0.03을 오래 고정 유지한 게 오히려
     이 폭주를 부추긴 원인.
  대응(③에 적용): V8과 무관하게 "붕괴/폭주 메커니즘 자체"를 겨냥한 장치들 —
    1) LogStdClampCallback: 매 롤아웃 시작 시 policy.log_std를 [--log-std-min,
       --log-std-max](기본 -3.0~1.0, std≈0.05~2.7)로 강제 clamp — 폭주 방지의 핵심.
       평가는 항상 deterministic=True(평균만 사용)라 log_std 범위가 최종 판단력을
       제약하지 않음.
    2) ent_coef 기본값을 0.03→0.01, 종료값 0.01→0.005로 하향 (연속 액션엔 이산 때보다
       약한 압력이 적절 — 무제한 상방 엔트로피에 큰 계수를 오래 주면 폭주 유인이 됨).
    3) ent_coef를 --ent-coef-hold-frac(기본 0.7) 지점까지 고정 유지 후 감쇠 (유지).
    4) explore_bonus (유지): Enter 시 실현손익과 별개로 붙는 임시 보너스.
       --explore-bonus-start(기본 0.05)에서 --explore-bonus-decay-frac(기본 0.5)
       지점까지 선형으로 0에 수렴. cum_pnl/평가 점수엔 전혀 반영 안 됨.
  ③ 적용 후 재확인(같은 seed0, ③ 설정으로 재시작): log_std 폭주는 완전히 해소(entropy_loss
     -10.7 근처에서 안정). 그러나 150만~250만 스텝 구간에 BTC가 3회 연속 거래 0건으로
     재하락, 학습 중 rollout ep_rew_mean도 -0.27→-0.35로 지속 악화 — 폭주와는 별개로
     원래의 콜드스타트 함정(진입=확실한 비용, Skip=항상 0)이 여전히 살아있음을 확인.
  대응(④, 2026-07-16 추가): explore_bonus만으로 못 뚫은 이유를 손익 스케일로 역산 —
    고레버리지(최대 100배)에서는 손절선이 강제청산가에 눌려(`max(SL, 청산가)`) 거의
    항상 최대손실(-100, reward 단위 -1.0)로 청산되기 쉬운데, 기존 explore_bonus=0.05는
    이 손실 1건에도 못 미쳐 무의미했음. 두 가지 추가:
    5) LeverageMaxSchedule: adaptive 모드 레버리지 상한을 --leverage-max-start(기본 10)
       에서 시작해 --leverage-curriculum-frac(기본 0.3) 지점까지 --leverage-max-full
       (기본 100)로 선형 확대. 초반 무작위 탐험의 손실 분산 자체를 구조적으로 줄임
       (인센티브가 아니라 탐험 공간 제한). 평가는 항상 leverage_max=100(전체 범위)인
       새 env를 쓰므로 최종 정책의 레버리지 선택 폭엔 영향 없음.
    6) explore_bonus_start 0.05→0.15로 상향 (더 이상 흔한 손실 규모에 곧바로 묻히지
       않도록).

사용법:
  python src/rl_v9/train_v9.py --seed 0                          # 기본 50M 스텝, exit_mode=adaptive
  python src/rl_v9/train_v9.py --exit-mode rl ...                # 풀 컨트롤 모드 (붕괴 이력 있음)
  python src/rl_v9/train_v9.py --timesteps 30000 --workers 2 --dummy-vec --cache-suffix _recent120d  # 스모크
"""
import os
import sys
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import TradingEnvV9  # noqa: E402
from eval import (cache_path_for, split_bounds, run_policy_on_range, compute_metrics,  # noqa: E402
                  compound_metrics, monthly_sel_score, MIN_TRADES_PER_MONTH)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

LR_START = 3e-4


def mode_subdir(exit_mode):
    """2026-07-20: 모델/로그 산출물을 exit_mode별 폴더로 분리 — rl은 v9_fullctrl/,
    adaptive/rule은 v9_adaptive_ppo/ (rule은 adaptive와 기존에 파일명 접두사(v9_ppo_)를
    공유하던 관례를 그대로 이어받아 같은 폴더 사용). 세 모드 다 알고리즘은 PPO로 동일 —
    폴더명이 "fullctrl"인 이유는 mode_tag와 동일(train.py 상단 주석 참고)."""
    return "v9_fullctrl" if exit_mode == "rl" else "v9_adaptive_ppo"


def make_env_fn(cache_path, lo, hi, episode_len_rows, decision_stride, seed, env_kwargs):
    def _init():
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
    model_dir = os.path.join(MODEL_DIR, mode_subdir(args.exit_mode))
    os.makedirs(model_dir, exist_ok=True)

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

    class LogStdClampCallback(BaseCallback):
        """연속 액션(adaptive) 정책의 log_std는 SB3 기본값으로 상한이 없어, ent_coef가 계속
        엔트로피 보너스를 주면 표준편차가 무한정 커질 수 있음(2026-07-16 seed0에서 실측:
        entropy_loss -7→-62, 즉 표준편차가 수만 배로 폭주 — "확신을 갖고 굳어버림"이 아니라
        정반대인 "탐험폭 통제불능 폭주"였음이 SB3 소스로 부호 재확인 후 밝혀짐). 매 롤아웃
        시작 시(=매 model.train() 직후) log_std를 [min,max]로 강제 clamp. 평가는 항상
        deterministic=True(평균값만 사용)라 log_std 범위는 최종 판단력에 영향 없음."""
        def __init__(self, log_std_min, log_std_max):
            super().__init__()
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max

        def _on_rollout_start(self):
            import torch as th
            log_std = getattr(self.model.policy, "log_std", None)
            if log_std is not None:
                with th.no_grad():
                    log_std.data.clamp_(self.log_std_min, self.log_std_max)

        def _on_step(self):
            return True

    class ExploreBonusSchedule(BaseCallback):
        """adaptive 모드 전용: Enter 시 실현손익과 별개로 붙는 임시 탐험 보너스.
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

    class LeverageMaxSchedule(BaseCallback):
        """adaptive 모드 전용: 레버리지 상한을 --leverage-max-start에서 시작해
        --leverage-curriculum-frac 지점까지 --leverage-max-full로 선형 확대.
        레버리지가 높을수록 손절선이 강제청산가에 눌려 거의 항상 최대손실(-100)로
        청산되기 쉬워, 초반 무작위 탐험에서 손실 분산이 과도하게 커지는 것을 억제하는
        장치 (2026-07-16, explore_bonus만으로 콜드스타트가 안 뚫려 추가). 평가는 항상
        leverage_max=LEVERAGE_RANGE[1](기본값)인 새 env를 쓰므로 최종 판단력의 레버리지
        선택 폭엔 영향 없음 — 학습 중 탐험 분산만 조절. env_method 필요 이유는
        ExploreBonusSchedule 독스트링 참고 (set_attr 그림자 속성 버그).
        2026-07-17: LEVERAGE_RANGE 상한을 100→50으로 하향 — best 체크포인트 실측 결과
        근접청산(-95 이하) 거래 62건이 전부 레버리지 45배 이상, 89%가 정확히 100배,
        20배 이하는 0건이었음. 학습 재시작 필요(액션 매핑 자체가 바뀌므로 기존 체크포인트
        이어서 못 씀)."""
        def __init__(self, lev_start, lev_full, growth_timesteps):
            super().__init__()
            self.lev_start = lev_start
            self.lev_full = lev_full
            self.growth_timesteps = max(growth_timesteps, 1)

        def _on_rollout_start(self):
            frac = min(self.num_timesteps / self.growth_timesteps, 1.0)
            lev_max = self.lev_start + (self.lev_full - self.lev_start) * frac
            self.training_env.env_method("set_curriculum", leverage_max=float(lev_max))

        def _on_step(self):
            return True

    class ValidationCallback(BaseCallback):
        def __init__(self, eval_freq):
            super().__init__()
            self.eval_freq = eval_freq
            self.last_eval = 0
            self.best_score = -np.inf

        def _on_step(self):
            if self.num_timesteps - self.last_eval < self.eval_freq:
                return True
            self.last_eval = self.num_timesteps
            scores = {}
            btc_sel = None  # BTC 월별 log-multiple 선택 점수 (2026-07-20: 4분기 v9_score 대체)
            btc_logs_m = None
            btc_eligible = False
            for sym, path in cache_paths.items():
                lo, hi = bounds[path]["valid"]
                if hi - lo < 100:
                    continue
                trades = run_policy_on_range(
                    self.model, path, lo, hi,
                    n_segments=args.eval_segments, decision_stride=args.stride,
                    **env_kwargs,
                )
                ts = np.load(path)["ts_1m"]
                m = compute_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
                # 2026-07-18: 모델 선택 기준을 v8_score → v9_score로 교체
                # (top1 제외 total_pnl + win_rate×1000 − near_liq_pct×50000).
                # 2026-07-19: v8_score는 기록에서도 완전 폐기.
                scores[sym] = m["v9_score"]
                for key in ("trades", "trades_per_month", "win_rate", "total_pnl",
                            "max_single_loss", "pnl_std", "v9_score",
                            "near_liq_n", "near_liq_pct"):
                    self.logger.record(f"valid/{sym}/{key}", m[key])
                # 복리(전액 재투입) 지표 — 실전 운용 방식 기준 참고용 기록 (2026-07-19 추가).
                # 모델 선택에는 미사용 (선택 기준은 v9_score 4분할 평균−표준편차 그대로).
                cm = compound_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
                self.logger.record(f"valid/{sym}/compound_multiple", cm["multiple"])
                self.logger.record(f"valid/{sym}/compound_mdd_pct", cm["mdd_pct"])
                if sym == "BTC-USDT-SWAP":
                    # 2026-07-20: 모델 선택 기준을 "4분기 v9_score 평균−표준편차" →
                    # "월별 복리 log-multiple 평균−표준편차"로 교체. 근거(0719-1459 런 실측):
                    # 승률×1000 항이 점수를 지배해 total_pnl +28(사실상 본전, top1 제거 시
                    # 적자)인 9.5M이 +418인 50M을 제치고 best로 뽑혔고, 실전 운용 방식(전액
                    # 재투입 복리)과 선택 목적이 어긋나 복리 MDD 99% 체크포인트가 걸러지지
                    # 않았음. 월 12표본이라 분기 4표본보다 분산 추정도 안정적. 레짐 편중
                    # 감점(−std)이라는 취지는 그대로 계승. v9_score는 TB 기록으로만 유지.
                    btc_sel, btc_logs_m = monthly_sel_score(trades, int(ts[lo]), int(ts[hi - 1]))
                    # "무거래 = 월배수 1.0 = 중립"이 손실 정책보다 우대되는 함정 차단:
                    # 월평균 거래수 미달이면 best 후보 자격 자체를 박탈 (합격 기준 ①과 동일 문턱)
                    btc_eligible = m["trades_per_month"] >= MIN_TRADES_PER_MONTH
                if self.verbose:
                    print(f"[valid @{self.num_timesteps:,}] {sym}: "
                          f"trades={m['trades']} pnl={m['total_pnl']:+.1f} v9_score={m['v9_score']:+.1f}")
            # 모델 선택은 BTC 단독 (2026-07-19: min(BTC,ETH) 기준은 만성적으로 ETH에 끌려
            # 내려가 BTC에서 잘하는 체크포인트를 놓쳤음. ETH는 지표 기록만 유지하는 참고용).
            if btc_sel is not None:
                self.logger.record("valid/BTC-USDT-SWAP/sel_monthly_log", btc_sel)
                if len(scores) > 1:
                    self.logger.record("valid/min_v9_score", float(min(scores.values())))  # 참고용으로 계속 기록
                if btc_eligible and btc_sel > self.best_score:
                    self.best_score = btc_sel
                    path = os.path.join(model_dir, f"{run_name}_best")
                    self.model.save(path)
                    with open(path + "_info.json", "w", encoding="utf-8") as f:
                        json.dump({"timesteps": self.num_timesteps,
                                   "btc_sel_monthly_log": round(btc_sel, 4),
                                   "btc_monthly_multiples": [round(float(np.exp(l)), 3) for l in btc_logs_m]}, f)
                    if self.verbose:
                        print(f"[valid] new best (BTC monthly-log sel {btc_sel:+.4f}) -> {path}.zip")
            return True

    callbacks = [EntCoefSchedule(), ValidationCallback(args.eval_freq)]
    if args.exit_mode == "adaptive":
        # log_std 클램프는 adaptive 전용 개념 — rl 모드는 Discrete(이산) 행동이라 log_std
        # 자체가 없음(폭주 위험 없음).
        callbacks.append(LogStdClampCallback(args.log_std_min, args.log_std_max))
        # 2026-07-20: --leverage가 주어지면 "고정 상한 베이스라인" 모드 — 커리큘럼(학습 중
        # 탐험 분산 억제용, 평가는 항상 무시)이 아니라 학습/검증/평가 전체에 동일 상한을
        # 관통시키는 게 목적이라 커리큘럼 램프를 끄고 env_kwargs["leverage_max"]를 상수로 고정
        # (아래 env_kwargs 구성부 참고). --leverage 미지정 시엔 기존 커리큘럼 그대로 유지.
        if args.leverage is None and args.leverage_max_start < args.leverage_max_full:
            callbacks.append(LeverageMaxSchedule(
                args.leverage_max_start, args.leverage_max_full,
                args.leverage_curriculum_frac * args.timesteps,
            ))
    if args.exit_mode in ("adaptive", "rl") and args.explore_bonus_start > 0:
        # 2026-07-20: rl 모드에도 explore_bonus 이식 — 06-16 붕괴(진입=확실한 비용,
        # Hold=항상 0의 구조적 비대칭)에 대한 동일한 대응. env._step_rl의 Enter 액션이
        # 이 값을 읽는다.
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
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--workers", type=int, default=7)
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
    parser.add_argument("--eval-segments", type=int, default=1)  # 2026-07-19: 세그먼트 분할 폐기 (경계 오차)
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    parser.add_argument("--ent-coef-start", type=float, default=0.01,
                        help="탐험 강도 초기값 (기본 0.01. 2026-07-16: adaptive 모드는 연속 액션이라 엔트로피에 "
                             "상한이 없어 0.03은 log_std 폭주를 유발했음 — log_std 클램프 추가와 함께 하향)")
    parser.add_argument("--ent-coef-end", type=float, default=0.005,
                        help="탐험 강도 종료값 (기본 0.005)")
    parser.add_argument("--ent-coef-hold-frac", type=float, default=0.7,
                        help="전체 스텝 중 이 비율 지점까지 ent_coef_start를 그대로 유지 후 감쇠 시작 (기본 0.7. "
                             "2026-07-21: 0.85로 상향 시도했으나 같은 런에서 explore_bonus(0.15, 50%까지 유지)가 "
                             "과매매 붕괴를 오히려 25M까지 더 길게 끌고 간 정황이 드러나 원인이 ent_coef가 아닐 "
                             "가능성이 높아져 0.7로 원복 — explore_bonus 쪽을 먼저 조정해보기로 함)")
    parser.add_argument("--log-std-min", type=float, default=-3.0,
                        help="adaptive 모드 정책 log_std 하한 (기본 -3.0, std≈0.05). 매 롤아웃 시작 시 강제 clamp")
    parser.add_argument("--log-std-max", type=float, default=1.0,
                        help="adaptive 모드 정책 log_std 상한 (기본 1.0, std≈2.7). 폭주 방지의 핵심 안전장치")
    parser.add_argument("--explore-bonus-start", type=float, default=0.001,
                        help="adaptive/rl 공통, Enter 시 붙는 임시 탐험 보너스 초기값 (0이면 비활성화). "
                             "2026-07-16: 0.05는 adaptive의 고레버리지발 최대손실(-1.0 reward 단위)에 묻혀 "
                             "무효했음 — 0.15로 상향. 2026-07-21: rl 모드(레버리지 낮음, 기본 1)에서는 이 0.15가 "
                             "거꾸로 진입 시 확정 수수료비용(leverage×fee_rate, 레버리지1 기준 0.0005)의 300배에 "
                             "달해 실제 손익 신호를 덮어버림 — '진입을 자주'만 배우고 '잘'은 못 배우게 만듦. "
                             "0.001(수수료의 약 2배)로 재하향한 런(0722-0702)이 이 세션 rl 모드 최초로 valid+test "
                             "동시 통과 — 신규 기본값으로 확정 (6장 이력 참고)")
    parser.add_argument("--explore-bonus-decay-frac", type=float, default=0.5,
                        help="전체 스텝 중 이 비율 지점에서 탐험 보너스가 0으로 수렴 (기본 0.5)")
    parser.add_argument("--leverage-max-start", type=float, default=10,
                        help="adaptive 모드 레버리지 상한 커리큘럼 시작값 (기본 10). "
                             "2026-07-16: 초반 무작위 탐험이 100배 근처를 자주 뽑아 손절선이 강제청산가에 "
                             "눌리며 거의 항상 최대손실로 청산되는 게 콜드스타트를 더 어렵게 한 것으로 판단해 추가")
    parser.add_argument("--leverage-max-full", type=float, default=50,
                        help="레버리지 상한 커리큘럼 최종값 (기본 50, LEVERAGE_RANGE 상한과 일치. "
                             "2026-07-17: 100→50 하향, 근접청산 실측 근거는 LeverageMaxSchedule 독스트링 참고)")
    parser.add_argument("--leverage-curriculum-frac", type=float, default=0.3,
                        help="전체 스텝 중 이 비율 지점에서 레버리지 상한이 leverage-max-full에 도달 (기본 0.3)")
    parser.add_argument("--exit-mode", choices=["rl", "rule", "adaptive"], default="adaptive",
                        help="adaptive(기본)=진입 시 정책이 레버리지(1~100)/손절폭/반익절/완익절을 직접 결정, 본절이동 없음. "
                             "rule=V9 Design.md 9장 Fallback B(전역 고정 청산 파라미터, V8 엔진 그대로 재사용). "
                             "rl=보유 중 풀 컨트롤, 방향은 fade 고정·레버리지는 --leverage로 고정 "
                             "(2026-07-20 재설계 — Discrete(3) {Hold,Enter,Close}, 06-16 자유방향판 붕괴 이력은 env.py 참고)")
    parser.add_argument("--sl-multiplier", type=float, default=None, help="rule 모드 ATR 손절 배수 (기본 1.5, adaptive에선 무시)")
    parser.add_argument("--tp-half-level", type=float, default=None, help="rule 모드 반익절 레벨 (기본 0.30, adaptive에선 무시)")
    parser.add_argument("--be-trigger-level", type=float, default=None, help="rule 모드 본절이동 레벨 (기본 0.60, adaptive엔 없음)")
    parser.add_argument("--leverage", type=float, default=None,
                        help="레버리지 값 설정 (모드별 의미가 다름, 2026-07-20). "
                             "rule/rl: 고정 레버리지 상수(기본 20.0, env.py 생성자 기본값 사용). "
                             "adaptive: 정책이 고를 수 있는 레버리지 '상한'(leverage_max)을 이 값으로 고정 — "
                             "기존 --leverage-max-start/-full 커리큘럼(학습 중에만 적용, 평가는 항상 무시)을 "
                             "대체해 학습/검증/평가 전체에 동일 상한을 관통시킴 (예: --leverage 1로 '레버리지 "
                             "미사용' 베이스라인 구성). eval.py 평가 시 학습 때와 반드시 동일 값을 지정해야 함 — "
                             "rl 모드는 liq_dist 관측 피처가, adaptive는 실제 선택 가능 범위가 leverage에 의존.")
    parser.add_argument("--cache-suffix", default="", help="스모크용 캐시 suffix (예: _recent120d)")
    parser.add_argument("--dummy-vec", action="store_true", help="SubprocVecEnv 대신 DummyVecEnv (스모크/디버그)")
    args = parser.parse_args()

    env_kwargs = {"exit_mode": args.exit_mode}
    if args.exit_mode == "rule":
        for key, val in (("sl_multiplier", args.sl_multiplier), ("tp_half_level", args.tp_half_level),
                         ("be_trigger_level", args.be_trigger_level)):
            if val is not None:
                env_kwargs[key] = val
    if args.exit_mode in ("rule", "rl") and args.leverage is not None:
        env_kwargs["leverage"] = args.leverage
    if args.exit_mode == "adaptive" and args.leverage is not None:
        env_kwargs["leverage_max"] = args.leverage

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    cache_paths = {s: cache_path_for(s, args.cache_suffix) for s in args.symbols}
    for s, p in cache_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"cache not found for {s}: {p} — run prep_features_v9.py first")
    bounds = split_bounds(list(cache_paths.values()))

    episode_len_rows = args.episode_days * 1440 // args.stride * args.stride
    env_fns = []
    symbols = list(cache_paths.keys())
    for w in range(args.workers):
        sym = symbols[w % len(symbols)]  # 워커를 심볼별 균등 배분
        path = cache_paths[sym]
        lo, hi = bounds[path]["train"]
        env_fns.append(make_env_fn(path, lo, hi, episode_len_rows, args.stride, args.seed * 1000 + w, env_kwargs))

    vec_env = DummyVecEnv(env_fns) if args.dummy_vec else SubprocVecEnv(env_fns, start_method="spawn")

    # 2026-07-19: 런 이름에 시작 시각을 붙여 유니크화 — 재시작 시 TB 런 디렉토리와
    # models/ 체크포인트(best/final 포함)가 이전 런 산출물을 덮어쓰는 사고 방지.
    # 2026-07-20: 서버 로컬시간(UTC) 대신 KST로 표기.
    # 2026-07-20: rl 모드는 접두사를 v9_fullctrl_로 분리 — adaptive/rule(v9_ppo_) 산출물과
    # 모델/로그 파일명만으로 구분 가능하도록 (rl 모드는 관측/행동 공간이 달라 체크포인트
    # 호환 안 됨, 이름부터 명확히 갈라둠). 태그를 "rl"로 하지 않은 이유: 세 모드 전부
    # 알고리즘은 PPO로 동일한데 "ppo" 옆에 "rl"을 나란히 두면 마치 다른 알고리즘처럼
    # 오해를 사기 때문 — fullctrl(보유 중 풀 컨트롤)로 "환경 모드" 차이임을 명확히 함.
    # TB 로그는 파일명 접두사뿐 아니라 폴더 자체를 분리 — rl은 logs/v9_fullctrl/,
    # adaptive/rule은 logs/v9_adaptive_ppo/ (rule은 adaptive와 기존에 v9_ppo_ 접두사를
    # 공유하던 관례를 그대로 이어받아 같은 폴더). TensorBoard --logdir는 하위 폴더를
    # 재귀 스캔하므로(logs/rl_v9/... 구조로 기존에도 확인됨) 별도 설정 불필요.
    mode_tag = "fullctrl" if args.exit_mode == "rl" else "ppo"
    log_subdir = mode_subdir(args.exit_mode)
    run_name = f"v9_{mode_tag}_seed{args.seed}_{datetime.now(ZoneInfo('Asia/Seoul')).strftime('%m%d-%H%M')}"
    print(f"run_name: {run_name}")
    model = PPO(
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

    final_path = os.path.join(MODEL_DIR, log_subdir, f"{run_name}_final")
    model.save(final_path)
    print(f"final model saved: {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
