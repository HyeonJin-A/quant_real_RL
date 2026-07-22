"""
V9 평가 하네스 (V9 Design.md 8장)

- 결정론적 정책으로 분할 구간(train/valid/test) 전체를 롤아웃하고 거래 목록을 수집
- 지표: 거래수, 월평균 거래수, 승률, 총 PnL, 평균 PnL/거래, 최대 단일 손실(MSL),
  PnL 표준편차, Profit Factor, Top-1 수익 비중, 누적 PnL 곡선 MDD, V8 Score
- 사전 확정 합격 기준 ①~③ 자동 판정 (④는 --baseline-score 입력 시)
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

MIN_TRADES_PER_MONTH = 10.0    # 합격 기준 ① (2026-07-21: 3→10, 거래횟수 자체는 sel_monthly_log
                                # 점수에 반영되지 않아 저거래 노이즈 체크포인트가 게이트만 통과하면
                                # best가 될 수 있었음 — 문턱을 높여 저거래 구간을 더 확실히 차단)
MAX_TOP1_SHARE = 0.30          # 합격 기준 ②
NEAR_LIQUIDATION_THRESHOLD = -95.0  # 증거금(100) 거의 다 날린 거래 판정 기준 (관측 전용, 학습/점수 미반영)


def cache_path_for(symbol, suffix=""):
    # 2026-07-20: v9b — 타이밍 피처 4종 추가판. 구 v9 캐시는 구 체크포인트 평가용으로 보존
    # (단, 구 체크포인트 평가는 당시 코드(git) 기준으로만 유효 — 관측 차원이 다름)
    return os.path.join(CACHE_DIR, f"features_v9b_{symbol}{suffix}.npy")


def split_bounds(cache_paths):
    """모든 심볼에 공통인 날짜 경계를 계산해 심볼별 행 인덱스 경계로 변환"""
    ts_arrays = {p: np.load(p)["ts_1m"] for p in cache_paths}
    t_lo = min(int(a[0]) for a in ts_arrays.values())
    t_hi = max(int(a[-1]) for a in ts_arrays.values())
    t_train_end = t_lo + (t_hi - t_lo) * SPLIT_TRAIN_END
    t_valid_end = t_lo + (t_hi - t_lo) * SPLIT_VALID_END
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


def run_policy_on_range(model, cache_path, lo, hi, n_segments=1, decision_stride=1, fee_rate=0.0005,
                        exit_mode="rule", sl_multiplier=None, tp_half_level=None,
                        be_trigger_level=None, max_hold_bars=None, leverage=None, leverage_max=None):
    """
    [lo, hi) 구간을 시간순으로 롤아웃. 기본 n_segments=1 = 세그먼트 분할 없음 (2026-07-19:
    16분할 병렬은 경계 강제정산 + 재수렴 구간 때문에 거래수 ~8%/pnl ~12% 오차가 실측돼 폐기 —
    실전과 완벽히 동일한 단일 연속 롤아웃이 기본). n_segments>1은 빠른 근사가 필요할 때만.

    leverage: rule/rl 모드 전용 고정 레버리지. leverage_max: adaptive 모드 전용 레버리지 상한
    (학습 때 --leverage로 고정 상한 베이스라인을 구성했다면 평가도 반드시 동일 값을 지정할 것
    — 기본값(전체 범위)과 다르면 정책이 학습 때와 다른 선택 폭/관측을 보게 됨, 2026-07-20).
    """
    env_kwargs = {"decision_stride": decision_stride, "fee_rate": fee_rate,
                  "fixed_full_range": True, "exit_mode": exit_mode}
    for key, val in (("sl_multiplier", sl_multiplier), ("tp_half_level", tp_half_level),
                     ("be_trigger_level", be_trigger_level), ("max_hold_bars", max_hold_bars),
                     ("leverage", leverage), ("leverage_max", leverage_max)):
        if val is not None:
            env_kwargs[key] = val

    n_segments = max(1, min(n_segments, (hi - lo) // 1000 or 1))
    edges = np.linspace(lo, hi, n_segments + 1, dtype=np.int64)
    envs = []
    for k in range(n_segments):
        if edges[k + 1] - edges[k] < 100:
            continue
        env = TradingEnvV9(cache_path, start_idx=int(edges[k]), end_idx=int(edges[k + 1]), **env_kwargs)
        envs.append(env)

    obs_list = [env.reset(seed=0)[0] for env in envs]
    done = [False] * len(envs)
    while not all(done):
        active = [k for k, d in enumerate(done) if not d]
        obs_batch = np.stack([obs_list[k] for k in active])
        actions, _ = model.predict(obs_batch, deterministic=True)
        for a_idx, k in enumerate(active):
            # Discrete(rl/rule)면 스칼라, Box(adaptive)면 6차원 배열 — 둘 다 그대로 전달
            obs, _, _, truncated, _ = envs[k].step(actions[a_idx])
            obs_list[k] = obs
            if truncated:
                done[k] = True

    trades = []
    for env in envs:
        trades.extend(env.trades)
    trades.sort(key=lambda t: t["entry_ts"])
    return trades


def v9_score(total_pnl, win_rate, top1_pnl=0.0):
    """2026-07-18: V8 Score를 대체하는 모델 선택 기준. 거래횟수는 이미 충분히 확보되고 있어
    제외. total_pnl 대신 top1(최대 단일 수익) 1건을 제외한 총수익을 사용 —
    합격 기준 ③(top1 제거 후 흑자)을 점수화한 것으로, "대박 한 방" 의존 체크포인트가
    고득점하는 것을 방지. median이 아닌 이 방식을 쓴 이유: 승률 ~23% 구조(잦은 손절 +
    소수 큰 익절)에선 median pnl이 항상 음수라 총수익과의 연결이 끊기기 때문.
    2026-07-19: near_liq_pct 페널티 항(×50000) 제거 — 손실 예산 기반 레버리지 상한
    (env_v9.MAX_TRADE_LOSS_PCT=20)이 근접청산(-95)을 구조적으로 불가능하게 만들어 항상 0인
    죽은 항이 됨. near_liq_n/pct 지표 자체는 울타리 검증용 경보기로 계속 기록한다(항상 0이어야
    정상, 0이 아니면 상한 공식 버그)."""
    return (total_pnl - max(top1_pnl, 0.0)) + win_rate * 1000.0


def compute_metrics(trades, ts_lo_ms, ts_hi_ms):
    months = max((ts_hi_ms - ts_lo_ms) / (30.44 * 86400 * 1000), 1e-9)
    if not trades:
        return {
            "trades": 0, "trades_per_month": 0.0, "win_rate": 0.0,
            "total_pnl": 0.0, "avg_pnl": 0.0, "max_single_loss": 0.0,
            "pnl_std": 0.0, "profit_factor": 0.0, "top1_share": 0.0,
            "mdd": 0.0, "v9_score": 0.0, "months": months,
            "near_liq_n": 0, "near_liq_pct": 0.0,
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    total = float(pnls.sum())
    msl = float(min(pnls.min(), 0.0))
    std = float(pnls.std())
    top1 = float(wins.max()) if len(wins) else 0.0
    cum = np.cumsum(pnls)
    mdd = float((np.maximum.accumulate(cum) - cum).max())
    # 증거금(100 USDT) 거의 다 날린 거래 건수 — 학습/V8 Score엔 관여하지 않는 순수 관측 지표.
    # "총 PnL이 좋아도 개별 거래에서 계좌를 거의 다 날리는 일이 실제로 몇 번이나 있었는가"를 직접 확인하기 위함.
    near_liq = int((pnls <= NEAR_LIQUIDATION_THRESHOLD).sum())
    win_rate = len(wins) / len(pnls)
    near_liq_pct = near_liq / len(pnls)
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
        "v9_score": v9_score(total, win_rate, top1_pnl=top1),
        "months": months,
        "near_liq_n": near_liq,
        "near_liq_pct": near_liq_pct,
    }


def compound_metrics(trades, ts_lo_ms, ts_hi_ms, start_equity=100.0):
    """복리 운용 시뮬레이션 (2026-07-19 추가, 평가 전용 — 학습 env/보상은 고정 100 USDT 유지).

    실전 운용 방식(시작 자본 100 USDT, 매 거래 전재산을 증거금으로 투입) 기준의 성장률.
    별도 재시뮬레이션 없이 고정 사이징 거래 목록에서 정확히 재구성 가능한 근거:
      - 손익/수수료가 증거금에 선형 비례 (pos_size = 증거금×lev/price) → 거래 수익률
        r = pnl/100은 증거금 규모 불변. 레버리지 상한(MAX_TRADE_LOSS_PCT)도 % 기준이라 동일.
      - semi-MDP라 심볼 내 거래는 겹치지 않음(청산까지 fast-forward) → 순차 재투자 가정이
        정확히 성립. (심볼 간에는 겹칠 수 있으므로 복리 지표는 심볼별로만 산출한다.)
    equity ≤ 0(파산)은 현행 adaptive에선 거래당 손실 ≤ 20%라 불가능하지만 rule/rl 모드
    (-100 가능) 방어용으로 처리한다.
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
            break
        peak = max(peak, equity)
        mdd_pct = max(mdd_pct, (1.0 - equity / peak) * 100.0)
    multiple = equity / start_equity
    return {
        "start_equity": start_equity,
        "final_equity": float(equity),
        "multiple": float(multiple),                # 최종자본 / 시작자본
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


def yearly_breakdown(trades):
    by_year = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    for t in trades:
        year = datetime.fromtimestamp(t["exit_ts"] / 1000, tz=timezone.utc).year
        by_year[year]["pnl"] += t["pnl"]
        by_year[year]["trades"] += 1
    return dict(sorted(by_year.items()))


def acceptance(metrics, trades, baseline_score=None):
    """사전 확정 합격 기준 (V9 Design.md 8장). ⑤(시드)는 시드별 리포트를 모아 별도 판정."""
    pnls = np.array([t["pnl"] for t in trades], dtype=np.float64) if trades else np.array([])
    top1_win = float(pnls[pnls > 0].max()) if len(pnls[pnls > 0]) else 0.0
    result = {
        "1_trades_per_month_ge_10": metrics["trades_per_month"] >= MIN_TRADES_PER_MONTH,
        "2_top1_share_le_30pct": (metrics["total_pnl"] > 0 and metrics["top1_share"] <= MAX_TOP1_SHARE),
        "3_profitable_without_top1": (metrics["total_pnl"] - top1_win) > 0,
    }
    if baseline_score is not None:
        # 2026-07-19: v8_score 완전 폐기 — 베이스라인 비교도 v9_score 기준으로 통일
        result["4_v9_score_ge_baseline"] = metrics["v9_score"] >= baseline_score
    return result


def fmt_metrics(m):
    return (
        f"  trades={m['trades']}  /month={m['trades_per_month']:.2f}  win={m['win_rate']*100:.1f}%\n"
        f"  total_pnl={m['total_pnl']:+.1f}  avg={m['avg_pnl']:+.2f}  MSL={m['max_single_loss']:+.1f}\n"
        f"  std={m['pnl_std']:.1f}  PF={m['profit_factor']:.2f}  top1_share={m['top1_share']*100:.1f}%\n"
        f"  MDD={m['mdd']:.1f}  V9_Score={m['v9_score']:+.1f}\n"
        f"  near_liquidation={m['near_liq_n']}건 ({m['near_liq_pct']*100:.1f}%, "
        f"pnl<={NEAR_LIQUIDATION_THRESHOLD:.0f} 기준)"
    )


def fmt_compound(cm):
    status = "  ⚠️ BANKRUPT" if cm["bankrupt"] else ""
    return (
        f"  [복리 100 USDT 전액재투입] final={cm['final_equity']:,.1f}  x{cm['multiple']:.2f}"
        f"  monthly=x{cm['monthly_growth']:.4f}  MDD={cm['mdd_pct']:.1f}%{status}"
    )


def evaluate(model, symbols, split, fee_rate=0.0005, decision_stride=1,
             n_segments=1, cache_suffix="", baseline_score=None, verbose=True,
             exit_mode="rule", sl_multiplier=None, tp_half_level=None,
             be_trigger_level=None, max_hold_bars=None, leverage=None, leverage_max=None):
    paths = {s: cache_path_for(s, cache_suffix) for s in symbols}
    bounds = split_bounds(list(paths.values()))
    report = {"split": split, "fee_rate": fee_rate, "exit_mode": exit_mode, "symbols": {}}
    all_trades = []
    for sym, path in paths.items():
        lo, hi = bounds[path][split]
        ts = np.load(path)["ts_1m"]
        if hi - lo < 100:
            print(f"[{sym}] split '{split}' too small, skipped")
            continue
        trades = run_policy_on_range(model, path, lo, hi, n_segments=n_segments,
                                     decision_stride=decision_stride, fee_rate=fee_rate,
                                     exit_mode=exit_mode, sl_multiplier=sl_multiplier,
                                     tp_half_level=tp_half_level, be_trigger_level=be_trigger_level,
                                     max_hold_bars=max_hold_bars, leverage=leverage, leverage_max=leverage_max)
        m = compute_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
        cm = compound_metrics(trades, int(ts[lo]), int(ts[hi - 1]))
        acc = acceptance(m, trades, baseline_score)
        report["symbols"][sym] = {
            "metrics": m, "compound": cm, "acceptance": acc, "yearly": yearly_breakdown(trades),
        }
        all_trades.extend(trades)
        if verbose:
            print(f"\n=== [{sym}] {split} ===")
            print(fmt_metrics(m))
            print(fmt_compound(cm))
            sel, logs_m = monthly_sel_score(trades, int(ts[lo]), int(ts[hi - 1]))
            print(f"  [월별 복리] sel={sel:+.4f} (mean−std of ln m)  "
                  f"m_i={[round(float(np.exp(l)), 2) for l in logs_m]}")
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
    parser.add_argument("--baseline-score", type=float, default=None,
                        help="비교 베이스라인의 V9 Score (합격 기준 ④ 판정용, 2026-07-19 v8_score 폐기)")
    parser.add_argument("--exit-mode", choices=["rl", "rule", "adaptive"], default="adaptive",
                        help="모델 학습 때 쓴 exit_mode와 반드시 일치해야 함 (adaptive=레버리지/손절/반익/완익 전부 정책이 결정, "
                             "rule=V9 Design.md 9장 Fallback B 전역 고정 청산, rl=보유 중 풀 컨트롤·방향 fade 고정)")
    parser.add_argument("--sl-multiplier", type=float, default=None)
    parser.add_argument("--tp-half-level", type=float, default=None)
    parser.add_argument("--be-trigger-level", type=float, default=None)
    parser.add_argument("--leverage", type=float, default=None,
                        help="rule/rl 모드 고정 레버리지. 학습 때 --leverage를 지정했다면 반드시 동일 값 지정 "
                             "(rl 모드는 liq_dist 관측 피처가 leverage에 의존하므로 불일치 시 평가가 왜곡됨)")
    parser.add_argument("--leverage-max", type=float, default=None,
                        help="adaptive 모드 레버리지 상한 베이스라인. 학습 때 train.py --leverage를 지정했다면 "
                             "반드시 동일 값 지정 (예: --leverage-max 1로 학습한 모델은 평가도 1로)")
    parser.add_argument("--out", default=None, help="리포트 JSON 저장 경로")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")

    if args.split == "test":
        print("⚠️  테스트셋 평가는 모든 개발이 끝난 뒤 단 1회만 수행해야 합니다 (V9 Design.md 8장).")

    report = evaluate(model, args.symbols, args.split, fee_rate=args.fee,
                      decision_stride=args.stride, n_segments=args.segments,
                      cache_suffix=args.cache_suffix, baseline_score=args.baseline_score,
                      exit_mode=args.exit_mode, sl_multiplier=args.sl_multiplier,
                      tp_half_level=args.tp_half_level, be_trigger_level=args.be_trigger_level,
                      leverage=args.leverage, leverage_max=args.leverage_max)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nreport saved: {args.out}")


if __name__ == "__main__":
    main()
