"""
V9 평가 하네스 (V9 Design.md 8장)

- 결정론적 정책으로 분할 구간(train/valid/test) 전체를 롤아웃하고 거래 목록을 수집
- 지표: 거래수, 월평균 거래수, 승률, 총 PnL, 평균 PnL/거래, 최대 단일 손실(MSL),
  PnL 표준편차, Profit Factor, Top-1 수익 비중, 누적 PnL 곡선 MDD
- 사전 확정 합격 기준 ①~③ 자동 판정 (④ v9_score 기준은 2026-07-25 v9_score 폐기와 함께 제거됨)
- 연도별 PnL 분해, 수수료 민감도(--fee)
- 심볼별 개별 + 합산 리포트 (합격 기준은 심볼별 각각 적용)

사용법:
  python src/rl_v9/eval_v9.py --model models/rl_v9/best_model.zip --split test
  python src/rl_v9/eval_v9.py --model ... --split test --fee 0.0007   # 수수료 민감도
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import TradingEnvV9, MARGIN_USDT  # noqa: E402

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT_DIR, "caches")

# 시계열 분할 (train 70% / valid 15% / test 15%) — 날짜 기준 공통 경계 (V9 Design.md 7장)
SPLIT_TRAIN_END = 0.70
SPLIT_VALID_END = 0.85
# 2026-07-23: 배포 전 최종학습용 분할 (train 95% / valid 5%, test 없음) — 개발 중 확정된 설정으로
# 데이터를 최대한 활용해 처음부터 재학습할 때만 사용 (--final-split). 이어학습 아님 — lr/ent_coef/
# explore_bonus 스케줄이 이미 소진된 기존 체크포인트에 이어붙이면 사실상 더 안 배움 (6장 이력 참고).
SPLIT_TRAIN_END_FINAL = 0.95
SPLIT_VALID_END_FINAL = 1.0

MIN_TRADES_PER_MONTH = 10.0    # 합격 기준 ① (2026-07-21: 3→10, 거래횟수 자체는 sel_monthly_log
                                # 점수에 반영되지 않아 저거래 노이즈 체크포인트가 게이트만 통과하면
                                # best가 될 수 있었음 — 문턱을 높여 저거래 구간을 더 확실히 차단)
MAX_TOP1_SHARE = 0.30          # 합격 기준 ②
NEAR_LIQUIDATION_THRESHOLD = -95.0  # 증거금(100) 거의 다 날린 거래 판정 기준 (관측 전용, 학습/점수 미반영)


def cache_path_for(symbol, suffix=""):
    # 2026-07-20: v9b — 타이밍 피처 4종 추가판. 구 v9 캐시는 구 체크포인트 평가용으로 보존
    # (단, 구 체크포인트 평가는 당시 코드(git) 기준으로만 유효 — 관측 차원이 다름)
    return os.path.join(CACHE_DIR, f"features_v9bx_{symbol}{suffix}.npy")


def split_bounds(cache_paths, final=False):
    """모든 심볼에 공통인 날짜 경계를 계산해 심볼별 행 인덱스 경계로 변환.
    final=True면 95/5(train/valid) 최종학습용 분할 사용 — test는 빈 구간이 됨."""
    train_end, valid_end = (SPLIT_TRAIN_END_FINAL, SPLIT_VALID_END_FINAL) if final \
        else (SPLIT_TRAIN_END, SPLIT_VALID_END)
    ts_arrays = {p: np.load(p)["ts_1m"] for p in cache_paths}
    t_lo = min(int(a[0]) for a in ts_arrays.values())
    t_hi = max(int(a[-1]) for a in ts_arrays.values())
    t_train_end = t_lo + (t_hi - t_lo) * train_end
    t_valid_end = t_lo + (t_hi - t_lo) * valid_end
    bounds = {}
    for p, ts in ts_arrays.items():
        i_train = int(np.searchsorted(ts, t_train_end))
        i_valid = int(np.searchsorted(ts, t_valid_end))
        bounds[p] = {
            "train": (0, i_train),
            "valid": (i_train, i_valid),
            "test": (i_valid, len(ts)),
        }
    return bounds


# 2026-07-28: 검증/평가 롤아웃 속도 최적화 (`검증 롤아웃 속도 최적화.md`).
# 실측(BTC valid 518,021행, torch 1스레드): model.predict() 124.8µs/step vs env.step() 1.4µs/step
# — 검증 시간의 98%가 "실제 연산"이 아니라 torch 호출 오버헤드(텐서 변환→디스패치→결과 반환)였음.
# [64,64] 초소형 MLP라 행렬 연산 자체는 마이크로초인데 프레임워크 왕복이 그 수십 배.
# torch predict는 batch=1/2/4가 119.5/129.5/124.6µs로 거의 평평 = 순수 고정 오버헤드라는 증거.
MASK_HUGE_NEG = np.float32(-1e8)  # sb3_contrib MaskableCategorical.apply_masking과 동일 상수


class NumpyDiscretePolicy:
    """검증/평가 롤아웃 전용 numpy 추론 래퍼 (Discrete 액션 + deterministic 전용).

    `model.predict(obs, deterministic=True[, action_masks=...])`와 동일한 시그니처/결과를
    제공하되 torch 호출 오버헤드를 제거한다(실측 124.8µs → 7.2µs, 약 17배).
    정책망이 Linear→Tanh→Linear→Tanh→action_net(Linear)뿐이라 numpy 3줄로 동일 연산이 되고,
    평가는 항상 deterministic=True(argmax)라 샘플링 경로도 불필요.

    ⚠️ 학습 중 검증이면 매 검증마다 새로 생성해야 한다 (정책 가중치가 계속 바뀌므로).
    구조가 조금이라도 다르면 절대 쓰지 말 것 — 생성은 `numpy_policy_if_supported()`로만.
    """

    def __init__(self, sb3_model):
        sd = {k: v.detach().cpu().numpy().astype(np.float32)
              for k, v in sb3_model.policy.state_dict().items()}
        # torch nn.Linear는 x @ W.T + b — 전치는 여기서 1회만 수행
        self.w = [sd["mlp_extractor.policy_net.0.weight"].T.copy(),
                  sd["mlp_extractor.policy_net.2.weight"].T.copy(),
                  sd["action_net.weight"].T.copy()]
        self.b = [sd["mlp_extractor.policy_net.0.bias"],
                  sd["mlp_extractor.policy_net.2.bias"],
                  sd["action_net.bias"]]

    def predict(self, obs_batch, deterministic=True, action_masks=None):
        if not deterministic:
            raise NotImplementedError("deterministic=True 전용 (평가 경로만 사용)")
        x = np.asarray(obs_batch, dtype=np.float32)
        x = np.tanh(x @ self.w[0] + self.b[0])
        x = np.tanh(x @ self.w[1] + self.b[1])
        logits = x @ self.w[2] + self.b[2]
        if action_masks is not None:
            # MaskableCategorical.apply_masking과 동일: 무효 행동 로짓을 -1e8로 치환.
            # sb3_contrib는 argmax(softmax(logits))를 쓰지만 softmax는 단조라 argmax는 동일.
            logits = np.where(action_masks, logits, MASK_HUGE_NEG)
        return np.argmax(logits, axis=-1), None


def numpy_policy_if_supported(model):
    """정책이 numpy로 정확히 재현 가능한 구조일 때만 NumpyDiscretePolicy를 반환, 아니면 None.

    None이면 호출부가 기존 torch `model.predict`로 폴백하므로 동작이 100% 불변이다.
    - net_arch/활성함수가 [64,64]+Tanh가 아니거나 features_extractor에 학습 파라미터가
      있으면(예: 향후 CNN/정규화 래퍼 도입) 즉시 폴백 — 조용히 틀린 결과를 내는 것 방지
    """
    try:
        import torch.nn as nn
        from gymnasium import spaces
        policy = model.policy
        if not isinstance(policy.action_space, spaces.Discrete):
            return None
        net = policy.mlp_extractor.policy_net
        if len(net) != 4 or not all(isinstance(net[i], t) for i, t in
                                    ((0, nn.Linear), (1, nn.Tanh), (2, nn.Linear), (3, nn.Tanh))):
            return None
        if not isinstance(policy.action_net, nn.Linear):
            return None
        if sum(p.numel() for p in policy.features_extractor.parameters()) != 0:
            return None
        return NumpyDiscretePolicy(model)
    except Exception:
        return None


def run_policy_on_ranges(model, targets, n_segments=1, decision_stride=1, fee_rate=0.0005,
                         max_hold_bars=None, leverage=None, use_numpy_policy=True):
    """여러 독립 레인(심볼)을 한 배치로 동시 롤아웃. targets: [(key, cache_path, lo, hi), ...]
    → {key: trades}

    각 레인은 여전히 자기 구간 전체를 1행씩 순차로 걷는다(실전과 동일한 단일 연속 롤아웃
    성질 보존). 바뀌는 것은 "한 번의 predict() 호출에 여러 심볼의 관측을 함께 넣는다"는 점뿐 —
    심볼끼리는 완전히 독립된 레인이라(BTC의 t시점 결정은 ETH 상태를 전혀 보지 않고, 상태 전이도
    자기 데이터에만 의존) 함께 추론하든 따로 하든 각 레인이 보는 정보·순서·결과가 동일하다.
    torch 호출 오버헤드가 배치 크기와 무관하게 고정이라, 심볼 2개면 호출 횟수가 절반이 된다.

    🚨 시간축 분할(n_segments>1)과 절대 혼동 금지 — 독립 레인 배치는 시간축을 안 건드려
    안전하지만, 한 심볼을 N토막 내는 것은 인위적 경계 강제정산 + 재수렴 구간이 생겨 결과가
    달라진다(2026-07-19 실측: 16분할에서 거래수 ~8%/pnl ~12% 오차, 폐기됨). n_segments는
    기존 인터페이스 호환용으로만 남겨두며 기본값 1(분할 없음)을 유지할 것.

    use_numpy_policy=False면 torch `model.predict` 경로를 강제한다 (A/B/C 파리티 검증용).
    """
    env_kwargs = {"decision_stride": decision_stride, "fee_rate": fee_rate,
                  "fixed_full_range": True}
    for key, val in (("max_hold_bars", max_hold_bars), ("leverage", leverage)):
        if val is not None:
            env_kwargs[key] = val

    lane_keys, envs = [], []
    for key, cache_path, lo, hi in targets:
        if hi - lo < 100:
            continue
        ns = max(1, min(n_segments, (hi - lo) // 1000 or 1))
        edges = np.linspace(lo, hi, ns + 1, dtype=np.int64)
        for k in range(ns):
            if edges[k + 1] - edges[k] < 100:
                continue
            envs.append(TradingEnvV9(cache_path, start_idx=int(edges[k]),
                                     end_idx=int(edges[k + 1]), **env_kwargs))
            lane_keys.append(key)

    # 2026-07-27: action masking(MaskablePPO) 지원 — 상태에 안 맞는 행동을
    # 물리적으로 제거한 채 학습됐으므로, 추론 시에도 동일하게 마스킹해야 학습 때와
    # 같은 유효 행동 후보로 argmax를 계산함 (`V9 Design TODO - Action Masking.md`).
    from sb3_contrib import MaskablePPO
    use_masking = isinstance(model, MaskablePPO)
    npolicy = numpy_policy_if_supported(model) if use_numpy_policy else None
    predictor = npolicy if npolicy is not None else model

    obs_list = [env.reset(seed=0)[0] for env in envs]
    done = [False] * len(envs)
    while not all(done):
        active = [k for k, d in enumerate(done) if not d]
        obs_batch = np.stack([obs_list[k] for k in active])
        if use_masking:
            mask_batch = np.stack([envs[k].action_masks() for k in active])
            actions, _ = predictor.predict(obs_batch, deterministic=True, action_masks=mask_batch)
        else:
            actions, _ = predictor.predict(obs_batch, deterministic=True)
        for a_idx, k in enumerate(active):
            obs, _, _, truncated, _ = envs[k].step(actions[a_idx])
            obs_list[k] = obs
            if truncated:
                done[k] = True

    out = {key: [] for key, _, _, _ in targets}
    for key, env in zip(lane_keys, envs):
        out[key].extend(env.trades)
    for key in out:
        out[key].sort(key=lambda t: t["entry_ts"])
    return out


def run_policy_on_range(model, cache_path, lo, hi, n_segments=1, decision_stride=1, fee_rate=0.0005,
                        max_hold_bars=None, leverage=None, use_numpy_policy=True):
    """단일 구간 롤아웃 — `run_policy_on_ranges`(복수형)의 1개 타깃 래퍼(기존 시그니처 유지).

    기본 n_segments=1 = 세그먼트 분할 없음 (2026-07-19: 16분할 병렬은 경계 강제정산 + 재수렴
    구간 때문에 거래수 ~8%/pnl ~12% 오차가 실측돼 폐기 — 실전과 완벽히 동일한 단일 연속
    롤아웃이 기본). n_segments>1은 빠른 근사가 필요할 때만.
    """
    return run_policy_on_ranges(
        model, [(0, cache_path, lo, hi)], n_segments=n_segments,
        decision_stride=decision_stride, fee_rate=fee_rate,
        max_hold_bars=max_hold_bars, leverage=leverage, use_numpy_policy=use_numpy_policy,
    )[0]


def compute_metrics(trades, ts_lo_ms, ts_hi_ms):
    months = max((ts_hi_ms - ts_lo_ms) / (30.44 * 86400 * 1000), 1e-9)
    if not trades:
        return {
            "trades": 0, "trades_per_month": 0.0, "win_rate": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0, "max_single_loss": 0.0,
            "pnl_std": 0.0, "profit_factor": 0.0, "top1_share": 0.0,
            "mdd": 0.0, "months": months,
            "near_liq_n": 0, "near_liq_pct": 0.0, "max_upnl": 0.0,
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    total = float(pnls.sum())
    msl = float(min(pnls.min(), 0.0))
    std = float(pnls.std())
    top1 = float(wins.max()) if len(wins) else 0.0
    # 2026-07-28 버그 수정: 시작점(누적 PnL 0)을 첫 고점으로 포함. 기존엔 np.maximum.accumulate가
    # cum[0]부터 시작해 **첫 거래부터 손실인 구간의 낙폭이 통째로 누락**됐다
    # (예: pnl=[-50,+100] → 실제 낙폭 50인데 0으로, [-30,-30,-30] → 90인데 60으로 보고).
    # 같은 파일의 compound_metrics는 peak=start_equity로 이미 시작점을 고점에 포함하고 있어
    # 두 MDD 계산이 서로 어긋나 있던 것. 이 mdd는 리포트 표시 전용(v9_kpi·합격기준·모델선택
    # 미사용)이고, 실측상 기존 런들은 초반 하락폭이 최대낙폭에 못 미쳐 값 변화가 없었음
    # (valid/test × BTC/ETH 4건 모두 차이 0.0) — 학습 초반 붕괴 체크포인트에서만 값이 달라진다.
    cum = np.concatenate(([0.0], np.cumsum(pnls)))
    mdd = float((np.maximum.accumulate(cum) - cum).max())
    # 증거금(100 USDT) 거의 다 날린 거래 건수 — 순수 관측 지표, 모델 선택엔 미관여.
    # "총 PnL이 좋아도 개별 거래에서 계좌를 거의 다 날리는 일이 실제로 몇 번이나 있었는가"를 직접 확인하기 위함.
    near_liq = int((pnls <= NEAR_LIQUIDATION_THRESHOLD).sum())
    win_rate = len(wins) / len(pnls)
    near_liq_pct = near_liq / len(pnls)
    # 2026-07-25: upnl_clip(관측 20번칸) 상한 재검토 근거 수집용 참고 지표.
    # env._close_position이 채워주는 거래별 "보유 중 관측된 최대 upnl/증거금 배수"(청산가/종가
    # 포함) 전체 중 최댓값 — 최종 실현 손익(top1)과 달리 "그 순간 관측값이 최대 몇 배까지
    # 갔었는가"를 담아, upnl_clip 상한을 몇으로 잡아야 극단값을 놓치지 않는지 판단하는 데 씀.
    max_upnl = max((t.get("max_upnl", 0.0) for t in trades), default=0.0)
    return {
        "trades": int(len(pnls)),
        "trades_per_month": len(pnls) / months,
        "win_rate": win_rate,
        "total_pnl": total,
        "avg_pnl": total / len(pnls),
        "max_single_loss": msl,
        "pnl_std": std,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "top1_share": top1 / total if total > 0 else float("inf"),
        "mdd": mdd,
        "months": months,
        "near_liq_n": near_liq,
        "near_liq_pct": near_liq_pct,
        "max_upnl": max_upnl,
    }


def compound_metrics(trades, ts_lo_ms, ts_hi_ms, start_equity=100.0):
    """복리 운용 시뮬레이션 (2026-07-19 추가, 평가 전용 — 학습 env/보상은 고정 100 USDT 유지).

    실전 운용 방식(시작 자본 100 USDT, 매 거래 전재산을 증거금으로 투입) 기준의 성장률.
    별도 재시뮬레이션 없이 고정 사이징 거래 목록에서 정확히 재구성 가능한 근거:
      - 손익/수수료가 증거금에 선형 비례 (pos_size = 증거금×lev/price) → 거래 수익률
        r = pnl/100은 증거금 규모 불변. 레버리지 상한(MAX_TRADE_LOSS_PCT)도 % 기준이라 동일.
      - semi-MDP라 심볼 내 거래는 겹치지 않음(청산까지 fast-forward) → 순차 재투자 가정이
        정확히 성립. (심볼 간에는 겹칠 수 있으므로 복리 지표는 심볼별로만 산출한다.)
    equity ≤ 0(파산)은 rl 모드에서 거래당 손실이 -100까지 가능하므로 방어용으로 처리한다.
    """
    months = max((ts_hi_ms - ts_lo_ms) / (30.44 * 86400 * 1000), 1e-9)
    equity = start_equity
    peak = start_equity
    mdd_pct = 0.0
    bankrupt = False
    for t in sorted(trades, key=lambda t: t["entry_ts"]):
        equity *= 1.0 + t["pnl"] / MARGIN_USDT
        if equity <= 0.0:
            equity = 0.0
            bankrupt = True
            # 2026-07-28 버그 수정: 기존엔 여기서 곧바로 break해 아래 mdd_pct 갱신을 건너뛰는
            # 바람에 **전액 파산이 mdd_pct=0.0(=낙폭 없음)으로 기록**됐다(첫 거래에서 파산하면
            # 0.0 그대로). 파산은 고점 대비 100% 손실이므로 명시적으로 100.0을 기록한다.
            # 지금까지는 compound_mdd_pct가 참고 지표라 표시상의 문제였지만, 2026-07-28부터
            # v9_kpi 점수(가중 30%) + 하드게이트(MDD<70)에 쓰이면서 방향이 정반대로 뒤집히는
            # 치명적 버그가 됨 — 파산 체크포인트가 z_mdd에서 최대 보너스(+4.12)를 받고 게이트도
            # 통과해버림. (실제 선택은 near_liq_n 가드가 간접 차단해왔으나 의존이 위태로웠음.)
            mdd_pct = 100.0
            break
        peak = max(peak, equity)
        mdd_pct = max(mdd_pct, (1.0 - equity / peak) * 100.0)
    multiple = equity / start_equity
    # 2026-07-28: 거래당 기하평균 수익률 — "이 수익률이 매 거래 똑같이 났다면 같은 최종 자본이
    # 나오는 값". 산술평균(avg_pnl)은 곱셈으로 쌓이는 자본을 재현하지 못한다(예: +50%,-50% →
    # 산술평균 0%지만 실제로는 100→150→75로 25% 손실, 기하평균 -13.4%가 정답).
    # 산술평균 ≥ 기하평균이 항상 성립하고 그 차이가 대략 분산의 절반이라(AM-GM),
    # 기하평균을 쓰면 변동성 손실(volatility drag)이 자동 반영된다.
    # multiple과 달리 **거래 횟수로 정규화**되므로 "많이 거래해서 배수가 커진 것"과
    # "거래당 엣지가 좋은 것"을 분리해서 볼 수 있다 (compound_multiple의 핵심 결함 보완).
    n_tr = len(trades)
    if n_tr == 0:
        geo = 0.0
    elif multiple <= 0.0:
        geo = -1.0          # 파산 — 유한 횟수로 0에 도달하는 일정 수익률은 -100%뿐
    else:
        geo = float(multiple ** (1.0 / n_tr) - 1.0)
    return {
        "start_equity": start_equity,
        "final_equity": float(equity),
        "multiple": float(multiple),                # 최종자본 / 시작자본
        "geo_mean_per_trade": geo,                  # 거래당 기하평균 수익률 (소수, 0.005 = +0.5%)
        "monthly_growth": float(multiple ** (1.0 / months)) if multiple > 0 else 0.0,  # 월평균 성장 배수
        "mdd_pct": float(mdd_pct),                  # 복리 자본곡선 고점 대비 최대 낙폭 %
        "bankrupt": bankrupt,
        "months": months,
    }


def monthly_log_multiples(trades, ts_lo_ms, ts_hi_ms, floor=0.05):
    """달력 월 단위 복리 배수의 log 목록 (2026-07-20, 모델 선택 기준용).

    각 월을 자본 1.0으로 리셋해 그 월의 거래(entry_ts 기준)만 복리 적용한 배수 m_i를
    구하고 ln(m_i)를 반환한다. mean(ln m_i)의 최대화는 전체 복리 성장률 최대화와 동치이고
    (log 합 = 전체 배수의 log), std(ln m_i)가 특정 레짐 편중을 감점한다 — 기존 v9_score
    4분기 선택식을 대체 (승률×1000 항이 점수를 지배해 "본전+균등" 체크포인트가 "수익+편중"
    을 이기는 문제와, 선택 목적이 실전 운용 방식(복리)과 어긋나는 문제를 함께 해소).

    - 무거래 월은 m=1.0 → log 0(중립)으로 포함. "아무것도 안 하는" 체크포인트가 이 점수로
      우대받는 것은 호출부의 월평균 거래수 가드(MIN_TRADES_PER_MONTH)가 차단한다.
    - floor: 월 자본 소멸 시 log → -inf 방지 클램프 (기본 0.05, log ≈ -3.0).
    """
    def ym(ms):
        d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return d.year * 12 + (d.month - 1)
    lo, hi = ym(ts_lo_ms), ym(ts_hi_ms)
    mult = {k: 1.0 for k in range(lo, hi + 1)}
    for t in trades:
        k = ym(t["entry_ts"])
        if k in mult:
            mult[k] *= 1.0 + t["pnl"] / MARGIN_USDT
    return [float(np.log(max(m, floor))) for _, m in sorted(mult.items())]


def monthly_sel_score(trades, ts_lo_ms, ts_hi_ms):
    """모델 선택 점수 = mean(ln m_i) − std(ln m_i). 월별 상세와 함께 반환."""
    logs = monthly_log_multiples(trades, ts_lo_ms, ts_hi_ms)
    arr = np.asarray(logs, dtype=np.float64)
    return float(arr.mean() - arr.std()), logs


# ---------------------------------------------------------------------------
# v9_kpi — best 체크포인트 선택 점수 (2026-07-28 도입, sel_monthly_log 단독 기준 대체)
#
# 안정성 최우선 요구에 따라 수익성(sel_monthly_log) + 낙폭(compound_mdd_pct) +
# 꼬리위험(max_single_loss) 세 축을 6:2:2로 합성한다 (2026-07-29: 4:3:3→6:2:2, 수익성
# 비중을 60%로 상향 — mdd:msl 상대 비율은 유지한 채 sel만 올림).
#
# ⚠️ 아래 상수는 절대 바꾸지 말 것 (바꾸면 과거 런과 점수 비교 불가):
#   ValidationCallback은 `v9_kpi > best_score`로 **학습 초반 점수와 후반 점수를 비교**해
#   best를 교체한다. 정규화 계수가 "지금까지 나온 표본의 통계"처럼 시간에 따라 변하면
#   초반/후반 점수가 서로 다른 자로 잰 값이 되어 대소 비교 자체가 무의미해진다.
#   그래서 러닝 z-score가 아니라 **고정 상수**를 쓴다.
#
# 스케일 산출 근거 (2026-07-28, 과거 4개 런 × BTC 검증 525회 중 하드게이트 통과분 실측):
#   pooled std — sel_monthly_log 0.516 / compound_mdd_pct 17.1 / max_single_loss 12.1
#   → 이 산포로 나눠야 세 항이 실제로 대등해진다. "값의 범위"(0~100 vs ±1.5)로 맞추면 틀림.
#   IQR 기준도 검토했으나 max_single_loss의 꼬리가 두꺼워(min -81.3) 혼자 분산의 50.7%를
#   차지해버려 기각. std 기준 + 4:3:3 가중 시 실측 분산기여 sel 37.0% / mdd 31.8% / msl 31.2%
#   (2026-07-29 6:2:2로 재조정, 분산기여 재실측은 아직 없음 — 가중치만 사용자 지시로 변경).
# 중심값(0 / 70 / -30)은 순위에 영향 없는 단순 오프셋 — v9_kpi≈0이 "평범한 체크포인트",
#   +1이 "종합 1σ 우수"로 읽히게 하는 해석 편의용.
# 주의: sel_monthly_log(−std 항)·compound_mdd_pct·max_single_loss는 서로 독립이 아니다
#   (실측 상관 sel↔mdd +0.76, sel↔msl +0.24, mdd↔msl +0.52). 6:2:2는 원래 4:3:3(세 축
#   균등이 아니라 [수익·낙폭] 축에 무게가 실린 구성)에서 수익성 비중을 한 단계 더 올린 것.
V9_KPI_CENTER = {"sel": 0.0, "mdd": 70.0, "msl": -30.0}
V9_KPI_SCALE = {"sel": 0.50, "mdd": 17.0, "msl": 12.0}
V9_KPI_WEIGHT = {"sel": 6.0, "mdd": 2.0, "msl": 2.0}
MAX_COMPOUND_MDD_PCT = 70.0   # best 후보 자격 하드게이트 (2026-07-28) — 이 이상은 실전 부적합


def v9_kpi(sel_monthly_log, compound_mdd_pct, max_single_loss, detail=False):
    """best 체크포인트 선택 점수 (높을수록 좋음). 세 지표 전부 "높을수록 좋음"으로 방향을
    맞춘 뒤 고정 상수로 정규화해 6:2:2 가중 평균한다 (위 상수 블록 주석 참고).

    compound_mdd_pct는 낮을수록, max_single_loss는 0에 가까울수록 좋으므로 부호를 뒤집는다.
    detail=True면 항별 기여도를 함께 반환 (진단용, 선택 로직엔 미사용).
    """
    z = {
        "sel": (float(sel_monthly_log) - V9_KPI_CENTER["sel"]) / V9_KPI_SCALE["sel"],
        "mdd": (V9_KPI_CENTER["mdd"] - float(compound_mdd_pct)) / V9_KPI_SCALE["mdd"],
        "msl": (float(max_single_loss) - V9_KPI_CENTER["msl"]) / V9_KPI_SCALE["msl"],
    }
    w_sum = sum(V9_KPI_WEIGHT.values())
    score = sum(V9_KPI_WEIGHT[k] * z[k] for k in z) / w_sum
    return (float(score), z) if detail else float(score)


def yearly_breakdown(trades):
    by_year = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    for t in trades:
        year = datetime.fromtimestamp(t["exit_ts"] / 1000, tz=timezone.utc).year
        by_year[year]["pnl"] += t["pnl"]
        by_year[year]["trades"] += 1
    return dict(sorted(by_year.items()))


def acceptance(metrics, trades):
    """사전 확정 합격 기준 (V9 Design.md 8장). ⑤(시드)는 시드별 리포트를 모아 별도 판정.
    2026-07-25: v9_score 폐기와 함께 구 ④(v9_score ≥ 베이스라인) 기준 제거 — 모델 선택은
    이미 sel_monthly_log로 일원화돼 있었고 v9_score는 TB 참고 기록 용도뿐이었음."""
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64) if trades else np.array([])
    top1_win = float(pnls[pnls > 0].max()) if len(pnls[pnls > 0]) else 0.0
    return {
        "1_trades_per_month_ge_10": metrics["trades_per_month"] >= MIN_TRADES_PER_MONTH,
        "2_top1_share_le_30pct": (metrics["total_pnl"] > 0 and metrics["top1_share"] <= MAX_TOP1_SHARE),
        "3_profitable_without_top1": (metrics["total_pnl"] - top1_win) > 0,
    }


def fmt_metrics(m):
    return (
        f"  trades={m['trades']}  /month={m['trades_per_month']:.2f}  win={m['win_rate']*100:.1f}%\n"
        f"  total_pnl={m['total_pnl']:+.1f}  avg={m['avg_pnl']:+.2f}  MSL={m['max_single_loss']:+.1f}\n"
        f"  std={m['pnl_std']:.1f}  PF={m['profit_factor']:.2f}  top1_share={m['top1_share']*100:.1f}%\n"
        f"  MDD={m['mdd']:.1f}\n"
        f"  near_liquidation={m['near_liq_n']}건 ({m['near_liq_pct']*100:.1f}%, "
        f"pnl<={NEAR_LIQUIDATION_THRESHOLD:.0f} 기준)\n"
        f"  max_upnl={m['max_upnl']:.2f}x (rl 전용 참고, upnl_clip 상한 재검토용)"
    )


def fmt_compound(cm):
    status = "  ⚠️ BANKRUPT" if cm["bankrupt"] else ""
    return (
        f"  [복리 100 USDT 전액재투입] final={cm['final_equity']:,.1f}  x{cm['multiple']:.2f}"
        f"  monthly=x{cm['monthly_growth']:.4f}  MDD={cm['mdd_pct']:.1f}%{status}\n"
        f"  거래당 기하평균={cm['geo_mean_per_trade']*100:+.4f}%/거래 "
        f"(거래 횟수 정규화 — multiple과 달리 '많이 거래해서 커진 것'과 무관한 순수 엣지)"
    )


def evaluate(model, symbols, split, fee_rate=0.0005, decision_stride=1,
             n_segments=1, cache_suffix="", verbose=True,
             max_hold_bars=None, leverage=None, final_split=False):
    paths = {s: cache_path_for(s, cache_suffix) for s in symbols}
    bounds = split_bounds(list(paths.values()), final=final_split)
    report = {"split": split, "fee_rate": fee_rate, "symbols": {}}
    all_trades = []
    # 2026-07-28: 심볼별 순차 롤아웃 → 독립 레인 배치 1회 호출 (속도 최적화, 결과 불변)
    targets = []
    for sym, path in paths.items():
        lo, hi = bounds[path][split]
        if hi - lo < 100:
            print(f"[{sym}] split '{split}' too small, skipped")
            continue
        targets.append((sym, path, lo, hi))
    trades_by_sym = run_policy_on_ranges(model, targets, n_segments=n_segments,
                                         decision_stride=decision_stride, fee_rate=fee_rate,
                                         max_hold_bars=max_hold_bars, leverage=leverage)
    for sym, path, lo, hi in targets:
        ts = np.load(path)["ts_1m"]
        trades = trades_by_sym[sym]
        m = compute_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
        cm = compound_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
        acc = acceptance(m, trades)
        sel, logs_m = monthly_sel_score(trades, int(ts[lo]), int(ts[hi - 1]))
        kpi, kpi_z = v9_kpi(sel, cm["mdd_pct"], m["max_single_loss"], detail=True)
        report["symbols"][sym] = {
            "metrics": m, "compound": cm, "acceptance": acc, "yearly": yearly_breakdown(trades),
            "sel_monthly_log": sel, "v9_kpi": kpi, "v9_kpi_z": kpi_z,
        }
        all_trades.extend(trades)
        if verbose:
            print(f"\n=== [{sym}] {split} ===")
            print(fmt_metrics(m))
            print(fmt_compound(cm))
            print(f"  [월별 복리] sel={sel:+.4f} (mean−std of ln m)  "
                  f"m_i={[round(float(np.exp(l)), 2) for l in logs_m]}")
            # best 선택 점수 — 학습 중 ValidationCallback이 쓰는 것과 완전히 동일한 식
            gate = "통과" if (m["trades_per_month"] >= MIN_TRADES_PER_MONTH
                            and m["near_liq_n"] == 0
                            and cm["mdd_pct"] < MAX_COMPOUND_MDD_PCT) else "탈락"
            print(f"  [v9_kpi] {kpi:+.4f}  (sel {kpi_z['sel']:+.2f} / mdd {kpi_z['mdd']:+.2f} / "
                  f"msl {kpi_z['msl']:+.2f}, 가중 6:2:2)  best후보 게이트: {gate}")
            print(f"  acceptance: {acc}")
            print(f"  yearly: { {y: round(v['pnl'], 1) for y, v in yearly_breakdown(trades).items()} }")

    if all_trades and len(report["symbols"]) > 1:
        all_trades.sort(key=lambda t: t["entry_ts"])
        ts_lo = min(t["entry_ts"] for t in all_trades)
        ts_hi = max(t["exit_ts"] for t in all_trades)
        m_all = compute_metrics(all_trades, ts_lo, ts_hi)
        report["combined"] = {"metrics": m_all}
        if verbose:
            print(f"\n=== [COMBINED] {split} ===")
            print(fmt_metrics(m_all))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--cache-suffix", default="", help="스모크용 캐시 suffix (예: _recent120d)")
    parser.add_argument("--leverage", type=float, default=None,
                        help="고정 레버리지. 학습 때 --leverage를 지정했다면 반드시 동일 값 지정 "
                             "(liq_dist 관측 피처가 leverage에 의존하므로 불일치 시 평가가 왜곡됨)")
    parser.add_argument("--out", default=None, help="리포트 JSON 저장 경로")
    parser.add_argument("--final-split", action="store_true",
                        help="배포 전 최종학습용 95/5(train/valid) 분할 기준으로 평가 (test 없음, 2026-07-23)")
    args = parser.parse_args()

    # 2026-07-27: action masking 도입으로 MaskablePPO 저장 포맷으로 전환됨 —
    # 기존(masking 이전) 체크포인트는 정책 클래스가 달라 호환 안 됨(재학습 필요).
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model, device="cpu")

    if args.split == "test":
        if args.final_split:
            raise SystemExit("--final-split엔 test 구간이 없습니다 (train 95%/valid 5%).")
        print("⚠️  테스트셋 평가는 모든 개발이 끝난 뒤 단 1회만 수행해야 합니다 (V9 Design.md 8장).")

    report = evaluate(model, args.symbols, args.split, fee_rate=args.fee,
                      decision_stride=args.stride, n_segments=args.segments,
                      cache_suffix=args.cache_suffix, leverage=args.leverage,
                      final_split=args.final_split)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nreport saved: {args.out}")


if __name__ == "__main__":
    main()
