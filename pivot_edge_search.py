"""
피봇 파라미터 (n, min_diff) 그리드 탐색 — Fade Edge t-통계량 기반

"찐저점/찐고점" 직관을 트레이드 시뮬레이션 없이 순수 가격 데이터로 직접 평가한다.
방법론 요약 (대화에서 확정된 스펙):

1. 각 피봇의 "확정 시점(T_confirm)"에서만 표본을 뽑는다.
   - t_peak(실제 극값 시점) 기준으로 재면 알고리즘 정의상 이미 반전이 보장된
     시점이라 look-ahead bias가 생긴다. T_confirm(= 거리 조건과 min_diff 조건이
     동시에 충족되어 피봇이 확정된 캔들)부터 측정해야 "이 피봇을 믿고 진입했을 때"를
     평가할 수 있다.
   - 🚨 2026-08-01 변경: 예전엔 원본을 건드리지 않으려고 이 파일에서 검출 로직을
     별도 재구현했었다. 그런데 원본(`algorithm.find_dynamic_pivots`)이 피봇 admission
     룩어헤드 수정으로 전면 재작성되면서 **확정 규칙 자체가 바뀌었고**(N-bar 극점
     검증 → 거리 `t−P>=n` AND 가격 `min_diff`의 단순 AND), `confirm_index`를 직접
     반환하게 됐다. 재구현본을 그대로 두면 실측 기준 피봇의 약 10%가 다른 봉에 잡히고
     확정 시점은 99.8%가 어긋나(중앙값 4봉) **탐색이 실제 쓰이는 정의와 다른 대상을
     최적화**하게 된다. 그래서 재구현을 폐기하고 원본을 직접 호출한다.

2. Fade Edge = -D_t * R[T_confirm -> T_confirm+M]
   - D_t: 직전 파동 방향 (고점 확정 = +1, 저점 확정 = -1)
   - R: 확정가 대비 M시간 뒤 종가 수익률
   - t-stat = mean(Fade Edge) / std(Fade Edge) * sqrt(N_eff)
   - N_eff: 확정 간격이 M(보유 horizon)보다 좁아 관측 윈도우가 겹치는 피봇들을
     그리디로 솎아낸(비중첩) 유효 표본 수. 명목 N을 그대로 쓰면 겹친 표본 때문에
     t값이 과대평가된다.

3. 보조 지표: MAE(최대 역행폭) — fade 방향의 반대(원래 파동 방향)로 얼마나
   더 밀렸는지를 측정. 단, V9는 레버리지를 고정하지 않고 정책이 매 진입마다
   1~50배 사이에서 직접 결정하므로(env_v9.py LEVERAGE_RANGE), "청산까지의
   거리"를 특정 레버리지로 환산해 하드 컷오프를 거는 것은 성립하지 않는다.
   MAE는 하드 필터가 아니라 **생존 조합끼리 비교하는 정보성 지표**로만 쓴다
   (t-stat이 비슷하면 MAE가 작은, 덜 휩쏘 타는 쪽을 우선 고려하는 용도).

3-b. 🚨 **train 구간만 사용** (2026-08-01 추가). 피봇 파라미터는 피처 정의의 일부이므로
   valid/test를 보고 고르면 그 자체가 누수다 — 이전 탐색(07-17)이 전 구간을 썼기 때문에
   "test에서도 성능이 안 떨어지는" 착시의 원인 중 하나로 지목됐다(design_spec 08-01 참고).
   `eval.split_bounds`와 동일한 날짜 경계(전 심볼 공통 t_lo~t_hi의 70%)로 잘라서 탐색한다.

4. 강건성 조건 — 아래를 모두 만족하는 (n, min_diff)만 "생존 조합"으로 채택:
   - 모든 서브기간(시간순 분할)에서 t_stat_adj > t-threshold
   - 여러 M(DEFAULT_M_HOURS_GRID 전반, 1~48h) 전반에서 준수 (특정 M에만 튀는 조합 배제)
   - BTC/ETH 양쪽에서 모두 성립 (V9 멀티심볼 일반화 원칙)
   - N_eff가 최소 표본수 이상
   (MAE는 조건에서 제외 — 위 3번 참고. 결과표에는 참고용으로 남긴다)

사용법:
  python src/param_search/pivot_edge_search.py
  python src/param_search/pivot_edge_search.py --symbols BTC-USDT-SWAP --recent-days 400   # 스모크 테스트
"""
from __future__ import annotations

import os
import sys
import argparse
import itertools

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "candle_data")
CACHE_DIR = os.path.join(ROOT_DIR, "caches")
OUT_DIR = os.path.join(ROOT_DIR, "pivot_edge_search")

sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from algorithm import find_dynamic_pivots  # noqa: E402  (v10 인과 검출기 — confirm_index 포함)

# train 경계: eval.split_bounds와 동일 (전 심볼 공통 t_lo~t_hi의 70%)
SPLIT_TRAIN_END = 0.70

DEFAULT_N_GRID = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]
DEFAULT_MIN_DIFF_GRID = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
DEFAULT_M_HOURS_GRID = [1, 2, 4, 6, 8, 10, 12, 16, 24, 36, 48]
DEFAULT_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
DEFAULT_SUBPERIODS = 4
DEFAULT_MIN_N_EFF = 30

DEFAULT_TIMEFRAME = "5m"
TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def load_candles(symbol: str, timeframe: str = DEFAULT_TIMEFRAME, recent_days: int | None = None) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    df = pd.read_csv(path, usecols=["ts", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])  # 일부 파일에 ts 결측 행이 섞여 있음 (예: BTC 15m 마지막 행)
    df = df.sort_values("ts").reset_index(drop=True)  # 파일은 내림차순 저장 -> 오름차순으로 정렬
    if recent_days is not None:
        cutoff = df["ts"].iloc[-1] - pd.Timedelta(days=recent_days)
        df = df[df["ts"] >= cutoff].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 피봇 확정 시점 추적 (src/Algorithm.py::find_dynamic_pivots와 동일 로직,
# 단 confirm_idx를 추가로 기록한다. 원본 파일은 수정하지 않는다.)
# ---------------------------------------------------------------------------
def find_pivots_with_confirmation(df: pd.DataFrame, n: int, min_diff: float):
    """v10 원본 검출기(`algorithm.find_dynamic_pivots`)를 그대로 호출하는 얇은 어댑터.

    🚨 여기서 로직을 재구현하지 말 것 — 탐색이 최적화하는 대상과 `prep_features`가
    실제로 쓰는 대상이 어긋나면 튜닝 결과가 무의미해진다. 원본이 이미 `confirm_index`
    (거리 조건과 min_diff 조건이 동시에 충족된 봉 = 그 시점 데이터만으로 판정 가능)를
    반환하므로 별도 지연 보정도 불필요하다.

    :return: list[dict(extreme_idx, confirm_idx, type, price)]
             (아직 확정되지 않은 마지막 후보는 원본이 이미 제외해서 반환한다)
    """
    return [{"extreme_idx": p["index"], "confirm_idx": p["confirm_index"],
             "type": p["type"], "price": p["price"]}
            for p in find_dynamic_pivots(df, n=n, min_diff=min_diff)]


def train_end_ts(symbols: list[str], timeframe: str) -> pd.Timestamp:
    """eval.split_bounds와 동일한 train 경계(전 심볼 공통 t_lo~t_hi의 70%)를 계산.

    피봇 파라미터는 피처 정의의 일부라 valid/test를 보고 고르면 누수다.
    경계 계산에만 전 심볼의 시작/끝 시각을 쓰고(값 자체를 보는 게 아님), 탐색은
    이 경계 이전 구간으로만 수행한다.
    """
    lo, hi = None, None
    for s in symbols:
        ts = pd.read_csv(os.path.join(DATA_DIR, f"{s}_{timeframe}.csv"), usecols=["ts"])["ts"]
        ts = pd.to_datetime(ts, errors="coerce").dropna()
        lo = ts.min() if lo is None else min(lo, ts.min())
        hi = ts.max() if hi is None else max(hi, ts.max())
    return lo + (hi - lo) * SPLIT_TRAIN_END


# ---------------------------------------------------------------------------
# Fade Edge / MAE 계산
# ---------------------------------------------------------------------------
def compute_fade_and_mae(pivots, close: np.ndarray, high: np.ndarray, low: np.ndarray, bars_m: int):
    """
    각 피봇(confirm_idx)에 대해 M(=bars_m 캔들)만큼 앞을 내다본 fade edge와 MAE를 계산.
    윈도우가 데이터 끝을 넘어가는(censored) 피봇은 제외.
    :return: list[dict(confirm_idx, type, fade, mae)]
    """
    L = len(close)
    out = []
    for p in pivots:
        ci = p["confirm_idx"]
        end = ci + bars_m
        if end >= L:
            continue
        confirm_price = close[ci]
        future_price = close[end]
        r = (future_price - confirm_price) / confirm_price
        d = 1 if p["type"] == "high" else -1
        fade = -d * r

        window_high = high[ci + 1:end + 1]
        window_low = low[ci + 1:end + 1]
        if d == 1:  # 고점 확정 -> fade는 숏 베팅 -> adverse는 추가 상승
            adverse = max(0.0, float(window_high.max()) - confirm_price) if len(window_high) else 0.0
        else:  # 저점 확정 -> fade는 롱 베팅 -> adverse는 추가 하락
            adverse = max(0.0, confirm_price - float(window_low.min())) if len(window_low) else 0.0
        mae = adverse / confirm_price

        out.append({"confirm_idx": ci, "type": p["type"], "fade": fade, "mae": mae})
    return out


def effective_n(records, bars_m: int) -> int:
    """확정 간격이 M(bars_m)보다 좁아 윈도우가 겹치는 표본을 그리디로 솎아낸 비중첩 표본 수."""
    if not records:
        return 0
    records_sorted = sorted(records, key=lambda r: r["confirm_idx"])
    kept = 1
    last_idx = records_sorted[0]["confirm_idx"]
    for r in records_sorted[1:]:
        if r["confirm_idx"] - last_idx >= bars_m:
            kept += 1
            last_idx = r["confirm_idx"]
    return kept


def summarize(records, bars_m: int) -> dict:
    n_obs = len(records)
    if n_obs == 0:
        return {"n_obs": 0, "n_eff": 0, "mean_fade": np.nan, "std_fade": np.nan,
                "t_stat_raw": np.nan, "t_stat_adj": np.nan, "mean_mae": np.nan}
    fade = np.array([r["fade"] for r in records])
    mae = np.array([r["mae"] for r in records])
    mean_fade = float(fade.mean())
    std_fade = float(fade.std(ddof=1)) if n_obs > 1 else np.nan
    n_eff = effective_n(records, bars_m)
    t_raw = mean_fade / std_fade * np.sqrt(n_obs) if std_fade and std_fade > 0 else np.nan
    t_adj = mean_fade / std_fade * np.sqrt(n_eff) if std_fade and std_fade > 0 and n_eff > 0 else np.nan
    return {
        "n_obs": n_obs, "n_eff": n_eff, "mean_fade": mean_fade, "std_fade": std_fade,
        "t_stat_raw": t_raw, "t_stat_adj": t_adj, "mean_mae": float(mae.mean()),
    }


def assign_subperiod(confirm_idx: int, boundaries: list[int]) -> int:
    for k, b in enumerate(boundaries):
        if confirm_idx < b:
            return k
    return len(boundaries)


# ---------------------------------------------------------------------------
# 메인 탐색 루프
# ---------------------------------------------------------------------------
def run_search(symbols, n_grid, min_diff_grid, m_hours_grid, n_subperiods,
                timeframe=DEFAULT_TIMEFRAME, recent_days=None):
    bar_minutes = TIMEFRAME_MINUTES[timeframe]
    rows = []

    t_train_end = train_end_ts(symbols, timeframe)
    print(f"train 경계(전 심볼 공통 {SPLIT_TRAIN_END:.0%}): ~ {t_train_end}  — 이후 구간은 탐색에서 제외\n")

    for symbol in symbols:
        print(f"[{symbol}] loading {timeframe} candles...")
        df = load_candles(symbol, timeframe=timeframe, recent_days=recent_days)
        n_all = len(df)
        df = df[df["ts"] < t_train_end].reset_index(drop=True)   # 🚨 train 구간만
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        L = len(df)
        boundaries = [int(L * (k + 1) / n_subperiods) for k in range(n_subperiods - 1)]
        print(f"[{symbol}] {L} bars (전체 {n_all} 중 train만), "
              f"{n_subperiods} subperiods, boundaries={boundaries}")

        for n, min_diff in itertools.product(n_grid, min_diff_grid):
            pivots = find_pivots_with_confirmation(df, n, min_diff)
            print(f"[{symbol}] n={n} min_diff={min_diff:.3f} -> {len(pivots)} pivots")
            if not pivots:
                continue

            for m_hours in m_hours_grid:
                bars_m = int(round(m_hours * 60 / bar_minutes))
                records = compute_fade_and_mae(pivots, close, high, low, bars_m)

                # 서브기간별로 분리 집계 + 전체(pooled) 집계
                buckets = {k: [] for k in range(n_subperiods)}
                for r in records:
                    buckets[assign_subperiod(r["confirm_idx"], boundaries)].append(r)

                for sub_k, sub_records in buckets.items():
                    stats = summarize(sub_records, bars_m)
                    rows.append({
                        "symbol": symbol, "n": n, "min_diff": min_diff,
                        "m_hours": m_hours, "subperiod": sub_k, **stats,
                    })

                stats_all = summarize(records, bars_m)
                rows.append({
                    "symbol": symbol, "n": n, "min_diff": min_diff,
                    "m_hours": m_hours, "subperiod": "all", **stats_all,
                })

    return pd.DataFrame(rows)


def rank_by_symbol(results: pd.DataFrame, min_n_eff: int) -> pd.DataFrame:
    """심볼별로 독립적인 (n, min_diff) 랭킹 (BTC/ETH를 하나로 묶어 요구하지 않음).

    이전 버전은 (symbol, subperiod, m_hours) 88개 셀 전부에서 t_stat_adj>threshold를
    강제하는 AND 필터였는데, 이건 과도하게 엄격했다:
      - 서브기간을 잘게 쪼갤수록 셀당 표본이 줄어 t-stat이 원래 더 노이즈해지는데
        그 좁은 표본 기준으로 유의성을 요구한 것
      - M(1~48h) 11개 전부가 동시에 유의미하길 요구한 것 — 서로 다른 성격의
        지평선인데 전부 통과하라는 건 무리한 요구
      - BTC/ETH를 한 조합으로 묶어 요구한 것

    그래서 아래처럼 완화한다:
      - 1차 지표는 **전체 히스토리 풀링**(subperiod=='all') t_stat_adj. 서브기간
        쪼개기보다 표본이 훨씬 크고 검정력이 높다. M별로 mean/worst를 리포트하되
        모든 M에서 통과를 요구하지 않고 분포로만 보여준다.
      - subperiod_sign_agreement: 서브기간별 mean_fade 부호가 풀링 부호와 일치하는
        비율. "통계적으로 유의미한가"가 아니라 "시간이 지나도 방향이 안 뒤집히는가"를
        보는 가벼운 정성적 체크(하드 필터 아님, 정렬 기준으로만 사용).
      - min_n_eff는 표본이 지나치게 적은 (n, min_diff)만 걸러내는 sanity filter
        (풀링된 큰 표본 기준이라 엄격하지 않음).
    """
    pooled = results[results["subperiod"] == "all"].copy()
    sub = results[results["subperiod"] != "all"].copy()

    out_frames = []
    for symbol in sorted(pooled["symbol"].unique()):
        pooled_sym = pooled[pooled["symbol"] == symbol]
        sub_sym = sub[sub["symbol"] == symbol]

        pooled_sign = pooled_sym.set_index(["n", "min_diff", "m_hours"])["mean_fade"].apply(np.sign)
        sub_merge = sub_sym.merge(
            pooled_sign.rename("pooled_sign").reset_index(),
            on=["n", "min_diff", "m_hours"], how="left",
        )
        sub_merge["agree"] = np.sign(sub_merge["mean_fade"]) == sub_merge["pooled_sign"]
        agreement = sub_merge.groupby(["n", "min_diff"])["agree"].mean().rename("subperiod_sign_agreement")

        grouped = pooled_sym.groupby(["n", "min_diff"]).agg(
            mean_t_stat_adj=("t_stat_adj", "mean"),
            worst_t_stat_adj=("t_stat_adj", "min"),
            best_t_stat_adj=("t_stat_adj", "max"),
            mean_mae=("mean_mae", "mean"),
            min_n_eff=("n_eff", "min"),
        ).reset_index()
        grouped = grouped.merge(agreement.reset_index(), on=["n", "min_diff"], how="left")
        grouped = grouped[grouped["min_n_eff"] >= min_n_eff].copy()
        grouped.insert(0, "symbol", symbol)
        out_frames.append(grouped)

    ranked = pd.concat(out_frames, ignore_index=True)
    # worst_t_stat_adj가 1차 정렬 기준: 높은 sign_agreement라도 "일관되게 음수"인
    # 경우가 있어(방향은 안정적이지만 그 방향이 fade 실패 쪽) agreement를 1차로
    # 쓰면 오도될 수 있다. worst_t_stat_adj를 우선하면 "모든 M에서 양수"인
    # 조합이 자연히 최상단에 온다.
    ranked = ranked.sort_values(
        ["symbol", "worst_t_stat_adj", "mean_t_stat_adj"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="피봇 (n, min_diff) 파라미터 Fade Edge t-통계량 그리드 탐색")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--n-grid", nargs="+", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--min-diff-grid", nargs="+", type=float, default=DEFAULT_MIN_DIFF_GRID)
    parser.add_argument("--m-hours-grid", nargs="+", type=float, default=DEFAULT_M_HOURS_GRID)
    parser.add_argument("--subperiods", type=int, default=DEFAULT_SUBPERIODS)
    parser.add_argument("--min-n-eff", type=int, default=DEFAULT_MIN_N_EFF)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, choices=sorted(TIMEFRAME_MINUTES),
                         help="캔들 timeframe (data/candle_data/{symbol}_{timeframe}.csv). 기본 5m")
    parser.add_argument("--recent-days", type=int, default=None, help="스모크 테스트용: 최근 N일만 사용")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--reuse-grid", action="store_true",
                         help="out-dir의 기존 grid_results.csv를 재사용해 랭킹만 다시 계산 "
                              "(피봇 탐지 재계산 생략, 그리드 옵션들은 무시됨)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "grid_results.csv")

    if args.reuse_grid:
        print(f"기존 그리드 재사용: {results_path}")
        results = pd.read_csv(results_path)
    else:
        results = run_search(
            symbols=args.symbols,
            n_grid=args.n_grid,
            min_diff_grid=args.min_diff_grid,
            m_hours_grid=args.m_hours_grid,
            n_subperiods=args.subperiods,
            timeframe=args.timeframe,
            recent_days=args.recent_days,
        )
        results.to_csv(results_path, index=False)
        print(f"\n전체 그리드 결과 저장: {results_path} ({len(results)} rows)")

    ranked = rank_by_symbol(results, min_n_eff=args.min_n_eff)
    ranked_path = os.path.join(args.out_dir, "survivors.csv")
    ranked.to_csv(ranked_path, index=False)

    print(f"\n심볼별 (n, min_diff) 랭킹 (worst_t_stat_adj -> mean_t_stat_adj 순 정렬, "
          f"min_n_eff>={args.min_n_eff} 필터, subperiod_sign_agreement/MAE는 참고 컬럼) 저장: {ranked_path}")
    for symbol in ranked["symbol"].unique():
        print(f"\n--- {symbol} top 10 ---")
        print(ranked[ranked["symbol"] == symbol].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
