"""
V9 피처 캐시 생성기 (V9 Design.md 3장, 4장)

- 심볼 파라미터화: BTC/ETH 등 심볼별로 features_v9_{SYMBOL}.npy 생성
- V8(prep_features_v7.py)의 인과성 보장 구조를 계승:
  5m 파동을 1m 스냅샷에 매핑, 매 1m 행은 그 시점에 알 수 있었던 정보만 포함
- V8 버그 수정: relative_volume_strength를 상수 0.0이 아닌 실제 계산값으로 기록
  (자기 이력 500개 5m 캔들 내 min-max 백분위 — src/Algorithm.py get_relative_volume_strength와 동일 정의)
- 추가 컬럼: adx_5m/atr_5m(직전 마감 5m 캔들, RMA 방식 = backtest_v8.calculate_adx와 동일),
  fib_pos(파동 내 되돌림 위치), pre_hit_0382(0.382 선터치 플래그, simulate_numba와 동일 추적),
  wave_age_min(신규 파동 이후 경과 분)
- 미사용 컬럼(VWAP/POC/SR)은 제거
- 심볼별 피처 분포 리포트(퍼센타일 표)를 함께 출력/저장 → 정규화 상수 검증용

사용법:
  python src/rl_v9/prep_features_v9.py --symbols BTC-USDT-SWAP ETH-USDT-SWAP
  python src/rl_v9/prep_features_v9.py --symbols BTC-USDT-SWAP --recent-days 120   # 스모크 테스트
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

from algorithm import find_dynamic_pivots, calc_rsi_divergence  # noqa: E402  (파동 감지 엔진은 V8과 동일한 것을 사용)

DATA_DIR = os.path.join(ROOT_DIR, "data", "candle_data")
CACHE_DIR = os.path.join(ROOT_DIR, "caches")

PIVOT_N = 8
PIVOT_MIN_DIFF = 0.02
# 2026-07-17: 심볼별 최적 (n, min_diff)를 src/param_search/pivot_edge_search로 별도 그리드서치해서 확정.
# 목록에 없는 심볼은 위 PIVOT_N/PIVOT_MIN_DIFF(V8 기본값)로 폴백.
PIVOT_PARAMS = {
    "BTC-USDT-SWAP": {"n": 3, "min_diff": 0.0075},
    "ETH-USDT-SWAP": {"n": 4, "min_diff": 0.015},
    # 2026-07-23: SOL은 별도 그리드서치 없이 ETH와 동일값 사용 (사용자 지정)
    "SOL-USDT-SWAP": {"n": 4, "min_diff": 0.015},
    # 2026-07-23: XRP는 아직 원본 데이터(data/candle_data/)가 없어 캐시는 미생성 — 나중에 데이터가
    # 준비되면 파라미터만 바로 쓸 수 있도록 등록만 해둠 (사용자 지정)
    "XRP-USDT-SWAP": {"n": 4, "min_diff": 0.04},
}
PRE_HIT_LEVEL = 0.382
VOL_STRENGTH_PERIOD = 500
WARMUP_5M = 200

CACHE_VER = "v9c"  # 2026-08-03: RSI 다이버전스 "last wave 외부 참조" 결함 수정
                   # (RSI-div-개편이전-핵심결함.md) — v9bx(버그①②만 수정) 위에 적용.
                   # 구버전은 last wave 끝(a_p1)과 전전 파동 끝(a_p3)을 비교(파동 바깥 참조).
                   # a_p2~a_p1(last wave 내부)에서 RSI 극값을 찾아 파동 끝 RSI와 비교하는
                   # 방식(algorithm.calc_rsi_divergence)으로 교체 — 임의 상수 없는 연속값,
                   # 이벤트 플래그 폐지. 컬럼 3종 개명: has_rsi_divergence/rsi_previous/
                   # divergence_price_gap_percent → div_rsi_gap/dist_from_div_peak_to_end/
                   # div_price_gap. v9bx와 비호환(스키마 변경).
                   #
                   # v9bx (2026-08-03): v9b에서 버그 2개만 수정 — 신규 피처/필터는 도입하지 않음.
                   # ① 다이버전스 (가격,RSI) 짝 불일치: 합성 극점 피봇 index를 실제 극점 봉으로
                   #    기록해 rsi2를 그 봉에서 읽음(forming 봉이 극점이면 판정 보류).
                   # ② 피봇 admission 룩어헤드(min_diff 가드 없음): find_dynamic_pivots를
                   #    confirm_index 기반 정순 인과 검출로 재작성(algorithm.py), 여기서도
                   #    admission을 confirm_index 기준으로 교체.

DTYPE_V9 = [
    ("i_1m", "i4"),
    ("i_5m", "i4"),
    ("ts_1m", "i8"),                         # ms
    ("high", "f4"),
    ("low", "f4"),
    ("close", "f4"),
    ("is_bullish", "i1"),                    # 마지막 파동 방향
    ("start_price", "f4"),
    ("end_price", "f4"),
    ("wave_scale_percent", "f4"),
    ("wave_duration_day", "f4"),
    # --- RSI 다이버전스 3종 (2026-08-03 재설계, algorithm.calc_rsi_divergence 참고) ---
    # 구버전의 (0/1 플래그 + rsi_previous + 가격갭) 조합을 대체. 발생/미발생 플래그가 없어져
    # "미발생 시 rsi_previous=50.0" 같은 센티넬 값과 불연속이 사라졌다 — 셋 다 상시 유효한 연속값.
    ("div_rsi_gap", "f4"),                   # 파동 내 RSI 극값 − 파동 끝 RSI (>=0). 0이면 확인(confirmation)
    ("dist_from_div_peak_to_end", "f4"),     # RSI 극점 → 파동 끝 (일 단위, wave_duration_day와 동일 단위/정규화)
    ("div_price_gap", "f4"),                 # RSI 극점 이후 가격이 추가로 이동한 폭 (%)
    ("relative_volume_strength", "f4"),      # 🚨 V8에서 0.0 하드코딩이던 버그 수정
    ("volatility_ratio", "f4"),
    ("adx_5m", "f4"),                        # 직전 마감 5m 캔들 기준
    ("atr_5m", "f4"),                        # 직전 마감 5m 캔들 기준 (RMA)
    ("fib_pos", "f4"),                       # (close-end)/(start-end): 0=파동끝, 1=파동시작
    ("pre_hit_0382", "i1"),
    ("wave_age_min", "f4"),
    # --- 2026-07-20 타이밍 피처 (승률 천장 ~27%의 원인이 "전환 시점 판정 정보 부재"라는
    # 진단에 따른 추가 — 60d 런 칼손절 즉사 분석 참고). 전부 과거 데이터만 사용(인과적). ---
    ("rsi_now", "f4"),                       # 직전 마감 5m 캔들의 RSI ("지금 과매수/과매도인가")
    ("ret_5m", "f4"),                        # close(t)/close(t-5분) - 1  (1m 종가 기준 트레일링 수익률)
    ("ret_15m", "f4"),                       # close(t)/close(t-15분) - 1
    ("ret_1h", "f4"),                        # close(t)/close(t-60분) - 1
]

REPORT_PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
REPORT_FIELDS = [
    "wave_scale_percent", "wave_duration_day", "div_rsi_gap",
    "dist_from_div_peak_to_end", "div_price_gap", "relative_volume_strength",
    "volatility_ratio", "adx_5m", "fib_pos", "wave_age_min",
    "rsi_now", "ret_5m", "ret_15m", "ret_1h",
]


def calculate_adx(df, period=14):
    """backtest_v8.calculate_adx와 동일한 RMA 방식 (V8 백테스트가 그라운드 트루스)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    def rma(series, window):
        alpha = 1 / window
        return pd.Series(series).ewm(alpha=alpha, adjust=False).mean()

    atr = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr
    minus_di = 100 * rma(minus_dm, period) / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = rma(dx, period)
    return adx.fillna(0.0).values, atr.fillna(0.0).values


def rolling_volume_strength(volumes, period=VOL_STRENGTH_PERIOD):
    """
    각 5m 캔들 볼륨의, 자기 시점까지의 트레일링 period개 윈도우 내 min-max 백분위.
    Algorithm.get_relative_volume_strength(내림차순 df, iloc[idx:idx+period])와 동일 정의를
    오름차순 배열에 벡터화한 것. 전부 과거 데이터만 사용하므로 인과적.
    """
    s = pd.Series(volumes)
    roll_min = s.rolling(period, min_periods=1).min().values
    roll_max = s.rolling(period, min_periods=1).max().values
    denom = roll_max - roll_min
    strength = np.where(denom > 0, (volumes - roll_min) / np.where(denom == 0, 1, denom), 0.0)
    return np.clip(strength, 0.0, 1.0)


def load_symbol(symbol, recent_days=None):
    df_5m = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_5m.csv"))
    df_1m = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_1m.csv"))
    for df in (df_5m, df_1m):
        if df["ts"].dtype in (np.int64, np.float64):
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="ms")
        else:
            df["ts_dt"] = pd.to_datetime(df["ts"])
    # OHLC 결측 행 제거 (BTC 1m 원본에 NaN 행 존재 확인됨 — NaN이 관측/보상에 전파되는 것 방지)
    df_5m = df_5m.dropna(subset=["open", "high", "low", "close"])
    df_1m = df_1m.dropna(subset=["open", "high", "low", "close"])
    df_5m = df_5m.sort_values("ts_dt").reset_index(drop=True)
    df_1m = df_1m.sort_values("ts_dt").reset_index(drop=True)

    if recent_days is not None:
        cutoff = df_1m["ts_dt"].iloc[-1] - pd.Timedelta(days=recent_days)
        df_5m = df_5m[df_5m["ts_dt"] >= cutoff].reset_index(drop=True)
        df_1m = df_1m[df_1m["ts_dt"] >= cutoff].reset_index(drop=True)
    return df_5m, df_1m


def build_cache(symbol, recent_days=None, n=None, min_diff=None):
    params = PIVOT_PARAMS.get(symbol, {"n": PIVOT_N, "min_diff": PIVOT_MIN_DIFF})
    n = params["n"] if n is None else n
    min_diff = params["min_diff"] if min_diff is None else min_diff
    print(f"[{symbol}] pivot params: n={n}, min_diff={min_diff}")
    t0 = time.time()
    print(f"\n=== [{symbol}] loading data ===")
    df_5m, df_1m = load_symbol(symbol, recent_days)
    total_5m = len(df_5m)
    total_1m = len(df_1m)
    print(f"5m rows: {total_5m:,} / 1m rows: {total_1m:,}")

    # --- 5m 사전계산 (전부 인과적 지표) ---
    adx_5m_arr, atr_5m_arr = calculate_adx(df_5m, 14)

    candle_range = df_5m["high"] - df_5m["low"]
    ambient_vol = candle_range.shift(1).rolling(10).mean()
    vol_ratio_arr = np.where(ambient_vol > 0, candle_range / ambient_vol, 1.0)

    vol_strength_arr = rolling_volume_strength(df_5m["volume"].values)

    if "rsi" not in df_5m.columns:
        df_5m["rsi"] = 50.0
    rsis_5m = df_5m["rsi"].fillna(50.0).values

    highs_5m = df_5m["high"].values
    lows_5m = df_5m["low"].values
    ts_5m_dt = df_5m["ts_dt"].values

    highs_1m = df_1m["high"].values
    lows_1m = df_1m["low"].values
    closes_1m = df_1m["close"].values
    ts_1m_dt = df_1m["ts_dt"].values
    # 🚨 반드시 ns 해상도를 경유해 epoch ms로 변환할 것 (2026-07-19 버그 수정).
    # pandas 3.x는 날짜 문자열을 datetime64[us]로 파싱하므로 astype(int64)의 단위가
    # 환경(pandas 버전)에 따라 ns/us로 달라짐 — us//10^6은 "초"가 되어 ts_1m 캐시가
    # 초 단위로 저장됐고, ms를 가정하는 wave_age_min(→ d5/d15 관측 게이트 전멸),
    # trades_per_month(1000배), yearly_breakdown(1970년) 등이 연쇄로 깨졌었음.
    ts_1m_ms = (df_1m["ts_dt"].astype("datetime64[ns]").astype("int64") // 10**6).values

    # --- 전역 피봇 (버그② 수정: 시간 오름차순 인과 검출) ---
    # 구버전은 내림차순 df에 검출기를 돌린 뒤 인덱스를 되돌렸다 — 대칭 N-Bar 판정을 역순에
    # 걸면 원래 시간축 기준으로는 오히려 확정을 더 앞당기는 결과가 됐다(algorithm.py 참고).
    # 정순 1회 순회로 검출하고 각 피봇의 confirm_index(확정 봉)를 함께 받는다.
    print("Calculating global pivots on 5m...")
    global_pivots = find_dynamic_pivots(df_5m, n=n, min_diff=min_diff)
    print(f"pivots: {len(global_pivots)}  ({time.time() - t0:.1f}s)")

    # --- 1m 순회: 파동 매핑 + 피처 추출 ---
    print("Mapping 1m rows to waves and extracting features...")
    idx_5m = 0
    pivot_ptr = 0
    forming_high = 0.0
    forming_low = 1e18

    # pre-hit / wave-age 추적 (simulate_numba :114-131과 동일 리셋 규칙)
    prev_start = -1.0
    prev_end = -1.0
    wave_filter_hit = False
    prev_wave_start = -1.0
    wave_born_ms = -1

    # 마감 캔들 구간 극값 메모 (idx_5m/피봇이 바뀔 때만 재계산)
    # 버그① 수정: 극값과 함께 극점 봉 인덱스도 추적 — 합성 피봇이 실제 극점 봉을 가리키게 해
    # (가격, RSI)를 같은 봉에서 읽는다
    memo_key = (-1, -1)
    memo_max_h = 0.0
    memo_min_l = 1e18
    memo_max_h_idx = -1
    memo_min_l_idx = -1

    # RSI 다이버전스 메모: (a_p2_idx, a_p1_idx)가 RSI 극점 탐색 구간을 완전히 결정하므로
    # 이 조합이 바뀔 때만 argmax를 다시 돌린다 (조합은 5m 봉 마감 빈도로만 바뀜).
    # 가격 갭은 파동 극점 가격(end_price)이 1m마다 갱신되므로 메모하지 않고 매 행 계산한다.
    div_key = (-1, -1)
    div_result = (0.0, 0, 0.0)

    rows = []
    t_loop = time.time()

    for i in range(total_1m):
        if i % 200000 == 0 and i > 0:
            print(f"  {i:,}/{total_1m:,}  ({time.time() - t_loop:.1f}s)")

        cur_ts = ts_1m_dt[i]
        while idx_5m + 1 < total_5m and cur_ts >= ts_5m_dt[idx_5m + 1]:
            idx_5m += 1
            forming_high = 0.0
            forming_low = 1e18

        if idx_5m < WARMUP_5M:
            continue

        if highs_1m[i] > forming_high:
            forming_high = highs_1m[i]
        if lows_1m[i] < forming_low:
            forming_low = lows_1m[i]

        # 버그② 수정 (admission 가드): 구버전은 `index <= idx_5m - n`으로 거리 조건만 보고
        # 가격 조건(min_diff)을 빼먹어, 반등이 min_diff에 도달하기 전에 피봇을 확정 처리했다
        # (실측 최대 47봉=3.9시간 미래 선행). 검출기가 두 조건을 모두 만족한 봉을
        # confirm_index로 알려주므로, 그 봉이 마감된 뒤에만 사용한다.
        while pivot_ptr < len(global_pivots) and global_pivots[pivot_ptr]["confirm_index"] < idx_5m:
            pivot_ptr += 1
        if pivot_ptr < 2:
            continue

        last_pivots_desc = global_pivots[max(0, pivot_ptr - 10):pivot_ptr][::-1]  # 최신순
        p1 = last_pivots_desc[0]
        p1_idx = p1["index"]

        # 확정 피봇 이후의 신규 극점(forming 포함) 추적 — V7/V8과 동일
        extreme_type = "high" if p1["type"] == "low" else "low"
        if p1_idx + 1 <= idx_5m - 1:
            if memo_key != (p1_idx, idx_5m):
                seg_h = highs_5m[p1_idx + 1:idx_5m]
                seg_l = lows_5m[p1_idx + 1:idx_5m]
                off_h = int(seg_h.argmax())
                off_l = int(seg_l.argmin())
                memo_max_h = seg_h[off_h]
                memo_min_l = seg_l[off_l]
                memo_max_h_idx = p1_idx + 1 + off_h
                memo_min_l_idx = p1_idx + 1 + off_l
                memo_key = (p1_idx, idx_5m)
            # 버그① 수정: forming 봉은 마감 구간 극값을 '초과'할 때만 극점(동률이면 RSI가
            # 존재하는 마감봉 우선). p1 자신이 극값이면 diff=0이라 어차피 합성 안 됨.
            if extreme_type == "high":
                extreme_price = max(p1["price"], memo_max_h, forming_high)
                if forming_high > max(memo_max_h, p1["price"]):
                    extreme_idx = idx_5m
                elif memo_max_h >= p1["price"]:
                    extreme_idx = memo_max_h_idx
                else:
                    extreme_idx = p1_idx
            else:
                extreme_price = min(p1["price"], memo_min_l, forming_low)
                if forming_low < min(memo_min_l, p1["price"]):
                    extreme_idx = idx_5m
                elif memo_min_l <= p1["price"]:
                    extreme_idx = memo_min_l_idx
                else:
                    extreme_idx = p1_idx
        else:
            if extreme_type == "high":
                extreme_price = max(p1["price"], forming_high)
                extreme_idx = idx_5m if forming_high > p1["price"] else p1_idx
            else:
                extreme_price = min(p1["price"], forming_low)
                extreme_idx = idx_5m if forming_low < p1["price"] else p1_idx

        diff_val = abs(p1["price"] - extreme_price)
        if diff_val >= min(p1["price"], extreme_price) * min_diff:
            # 버그① 수정: 합성 피봇 index를 idx_5m(현재 봉)이 아닌 실제 극점 봉으로 기록
            active_pivots = [{"index": extreme_idx, "type": extreme_type, "price": extreme_price}] + last_pivots_desc
        else:
            active_pivots = last_pivots_desc

        a_p1 = active_pivots[0]
        a_p2 = active_pivots[1]
        start_price = a_p2["price"]
        end_price = a_p1["price"]
        is_bullish = (a_p1["type"] == "high")

        wave_scale = abs(end_price - start_price) / start_price * 100
        wave_duration_day = max(0.001, (a_p1["index"] - a_p2["index"]) * 5 / (60 * 24))

        prev_5m_idx = max(0, idx_5m - 1)

        # RSI 다이버전스 (algorithm.calc_rsi_divergence — last wave 내부 모멘텀 천장 방식).
        # 2026-08-03 수정: a_p1이 지금 형성 중인 봉(아직 마감 전)이면, "직전 마감봉"으로
        # 클램프하는 게 아니라 그 이전에 확정돼 있던 진짜 극점 캔들로 되돌아간다. 저점/고점이
        # 실제로 갱신되기 전까지는 참조 RSI가 바뀌면 안 된다 — 시간이 흘러 마감봉이 하나 더
        # 생겼다는 이유만으로 참조 지점을 옮기면(구 버그) 파동 내내 값이 매 5분 계속
        # 표류하는 결과가 됨(실측 78% 행에서 stale 값 확인).
        if a_p1["index"] == idx_5m:
            if p1_idx + 1 <= idx_5m - 1:
                if extreme_type == "high":
                    div_end_idx = memo_max_h_idx if memo_max_h >= p1["price"] else p1_idx
                else:
                    div_end_idx = memo_min_l_idx if memo_min_l <= p1["price"] else p1_idx
            else:
                div_end_idx = p1_idx
        else:
            div_end_idx = a_p1["index"]

        div_pair_key = (a_p2["index"], div_end_idx)
        if div_pair_key != div_key:
            div_key = div_pair_key
            div_result = calc_rsi_divergence(
                rsis_5m, highs_5m, lows_5m,
                a_p2["index"], div_end_idx, is_bullish,
            )
        div_rsi_gap, div_dist_bars, div_ref_price = div_result
        dist_from_div_peak_to_end = div_dist_bars * 5 / (60 * 24)   # 봉 → 일 (wave_duration_day와 동일 단위)
        div_price_gap = (abs(end_price - div_ref_price) / div_ref_price * 100
                         if div_ref_price > 0 else 0.0)

        # 🚨 볼륨 상대강도 (버그 수정): 파동 끝 피봇 캔들 볼륨의 트레일링 500캔들 백분위.
        # 피봇이 현재 forming 캔들이면 직전 마감 캔들로 클램프하여 인과성 보장.
        vol_idx = min(a_p1["index"], prev_5m_idx)
        vol_strength = vol_strength_arr[vol_idx]

        # pre-hit 0.382 추적 (파동 start/end가 바뀌면 리셋 — simulate_numba와 동일)
        if start_price != prev_start or end_price != prev_end:
            prev_start = start_price
            prev_end = end_price
            wave_filter_hit = False
        dir_is_bullish = not is_bullish
        target_price = start_price * PRE_HIT_LEVEL + end_price * (1.0 - PRE_HIT_LEVEL)
        if not wave_filter_hit:
            if dir_is_bullish:
                if highs_1m[i] >= target_price:
                    wave_filter_hit = True
            else:
                if lows_1m[i] <= target_price:
                    wave_filter_hit = True

        # wave age: 파동 시작점(start_price)이 바뀌면 신규 파동으로 간주
        if start_price != prev_wave_start:
            prev_wave_start = start_price
            wave_born_ms = ts_1m_ms[i]
        wave_age_min = (ts_1m_ms[i] - wave_born_ms) / 60000.0

        denom = start_price - end_price
        fib_pos = (closes_1m[i] - end_price) / denom if denom != 0 else 0.0

        # 타이밍 피처: 트레일링 수익률 (1m 행 인덱스 기준 — 데이터 갭 시 근사, 드묾)
        ret_5m = closes_1m[i] / closes_1m[i - 5] - 1.0 if i >= 5 else 0.0
        ret_15m = closes_1m[i] / closes_1m[i - 15] - 1.0 if i >= 15 else 0.0
        ret_1h = closes_1m[i] / closes_1m[i - 60] - 1.0 if i >= 60 else 0.0

        rows.append((
            i, idx_5m, int(ts_1m_ms[i]),
            highs_1m[i], lows_1m[i], closes_1m[i],
            1 if is_bullish else 0,
            start_price, end_price,
            wave_scale, wave_duration_day,
            div_rsi_gap, dist_from_div_peak_to_end, div_price_gap,
            vol_strength, vol_ratio_arr[prev_5m_idx],
            adx_5m_arr[prev_5m_idx], atr_5m_arr[prev_5m_idx],
            fib_pos, 1 if wave_filter_hit else 0,
            wave_age_min,
            rsis_5m[prev_5m_idx],   # 직전 마감 5m 캔들 RSI (adx/atr와 동일 시점 관례)
            ret_5m, ret_15m, ret_1h,
        ))

    arr = np.array(rows, dtype=DTYPE_V9)
    os.makedirs(CACHE_DIR, exist_ok=True)
    suffix = f"_recent{recent_days}d" if recent_days else ""
    cache_path = os.path.join(CACHE_DIR, f"features_{CACHE_VER}_{symbol}{suffix}.npy")
    np.save(cache_path, arr)
    print(f"[{symbol}] cache saved: {cache_path}  rows={len(arr):,}  ({time.time() - t0:.1f}s)")

    report = distribution_report(arr, symbol)
    report_path = os.path.join(CACHE_DIR, f"dist_report_{CACHE_VER}_{symbol}{suffix}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[{symbol}] distribution report saved: {report_path}")
    return cache_path


def distribution_report(arr, symbol):
    """정규화 클립 범위가 이 심볼 분포를 커버하는지 검증하기 위한 퍼센타일 표 (V9 Design.md 4장)"""
    report = {"symbol": symbol, "rows": int(len(arr)), "percentiles": {}}
    header = f"{'feature':<34}" + "".join(f"p{p:<8}" for p in REPORT_PERCENTILES)
    print(f"\n--- [{symbol}] feature distribution ---")
    print(header)
    for field in REPORT_FIELDS:
        vals = arr[field].astype(np.float64)
        pcts = np.percentile(vals, REPORT_PERCENTILES)
        report["percentiles"][field] = {f"p{p}": round(float(v), 4) for p, v in zip(REPORT_PERCENTILES, pcts)}
        print(f"{field:<34}" + "".join(f"{v:<9.3f}" for v in pcts))
    # atr/close 비율(%)도 리포트 (관측에 쓰이는 실제 형태)
    atr_pct = arr["atr_5m"].astype(np.float64) / np.maximum(arr["close"].astype(np.float64), 1e-9) * 100
    pcts = np.percentile(atr_pct, REPORT_PERCENTILES)
    report["percentiles"]["atr_close_percent"] = {f"p{p}": round(float(v), 4) for p, v in zip(REPORT_PERCENTILES, pcts)}
    print(f"{'atr_close_percent':<34}" + "".join(f"{v:<9.3f}" for v in pcts))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    parser.add_argument("--recent-days", type=int, default=None,
                        help="최근 N일만 처리 (스모크 테스트용, 캐시 파일명에 suffix 부여)")
    args = parser.parse_args()
    for symbol in args.symbols:
        build_cache(symbol, recent_days=args.recent_days)


if __name__ == "__main__":
    main()
