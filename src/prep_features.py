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

from algorithm import find_dynamic_pivots  # noqa: E402  (파동 감지 엔진은 V8과 동일한 것을 사용)

DATA_DIR = os.path.join(ROOT_DIR, "data", "candle_data")
CACHE_DIR = os.path.join(ROOT_DIR, "caches")

PIVOT_N = 40  # 2026-08-05: 1분봉 네이티브 검출로 전환되며 5분봉 기준 그리드서치 값(8)에 x5 — 아래 참고
PIVOT_MIN_DIFF = 0.02
# 2026-07-17: 심볼별 최적 (n, min_diff)를 src/param_search/pivot_edge_search로 별도 그리드서치해서 확정.
# 목록에 없는 심볼은 위 PIVOT_N/PIVOT_MIN_DIFF(V8 기본값)로 폴백.
# 2026-08-05: 그리드서치는 5분봉 기준으로 했으므로(n=5분봉 캔들 개수), 피봇 검출이 1분봉
# 네이티브로 바뀐 지금은 n에 x5를 해서 저장 — 예전엔 호출부(find_dynamic_pivots)에서 매번
# n*5로 환산했는데, 상수 정의 시점에 미리 곱해두는 쪽으로 변경(호출부는 이제 n을 그대로 씀).
# min_diff는 가격 비율이라 스케일 불변, 그대로.
PIVOT_PARAMS = {
    "BTC-USDT-SWAP": {"n": 15, "min_diff": 0.0075},   # 5분봉 그리드서치 원값 n=3 x5
    "ETH-USDT-SWAP": {"n": 20, "min_diff": 0.015},    # 5분봉 그리드서치 원값 n=4 x5
    # 2026-07-23: SOL은 별도 그리드서치 없이 ETH와 동일값 사용 (사용자 지정)
    "SOL-USDT-SWAP": {"n": 20, "min_diff": 0.015},    # 5분봉 그리드서치 원값 n=4 x5
    # 2026-07-23: XRP는 아직 원본 데이터(data/candle_data/)가 없어 캐시는 미생성 — 나중에 데이터가
    # 준비되면 파라미터만 바로 쓸 수 있도록 등록만 해둠 (사용자 지정)
    "XRP-USDT-SWAP": {"n": 20, "min_diff": 0.04},     # 5분봉 그리드서치 원값 n=4 x5
}
PRE_HIT_LEVEL = 0.382
WARMUP_1M = 1000  # 2026-08-05: 피봇이 1분봉 네이티브가 되며 WARMUP_5M(200개=1000분)을
                  # 같은 실제 시간폭의 1분 단위로 이름·값 변경(200*5=1000)

CACHE_VER = "v10"   # 2026-08-05: v9d4에서 이름만 v10으로 변경(사용자 지정) — 앞으로 캐시
                    # 버전은 run_name(v10_maskablerl 등)과 동일한 네이밍을 따라간다. 값 변경
                    # 없음(features_v9d4_*.npy를 features_v10_*.npy로 파일명만 rename).
                    #
                    # v9d4 (2026-08-05): 5m 데이터 완전 제거 — 피봇(a_p1/a_p2) 검출 자체를 5분봉
                    # (df_5m/_5m.csv) 대신 1분봉(df_1m)에서 직접 수행. find_dynamic_pivots는
                    # 캔들 단위 무관 범용 함수라 무수정, 호출부는 df_1m/n(PIVOT_PARAMS에 이미
                    # 1분봉 기준 x5로 저장 — 최초엔 호출부에서 매번 n*5로 환산했다가, 상수
                    # 정의 시점에 미리 곱해두는 쪽으로 리팩터링. 결과값은 완전히 동일해 캐시
                    # 재생성 불필요)/min_diff(스케일 불변, 그대로)
                    # 로 교체. a_p1["index"]/a_p2["index"]가 이제 곧 1분 행
                    # 번호가 되어, A/B단계가 만들었던 "5분 인덱스 ↔ 1분 정밀 시각/인덱스"
                    # 변환 장치(ts_5m_high/low_precise_ms/idx1m, forming_*_ts_ms/idx1m,
                    # memo_max_h/min_l, idx_5m 경계루프) 전부 불필요해져 삭제 — 이제
                    # ts_1m_ms[a_p1["index"]]처럼 직접 조회. "형성중 5분봉 vs 마감된 5분봉"
                    # 이중 추적(forming_high/low + memo_max_h/min_l)도 p1(직전 확정 피봇)이
                    # 바뀔 때만 리셋되는 단일 running_extreme_price/idx로 통합(시드는 확정
                    # 지연 구간 포함 위해 p1_idx+1..현재 슬라이스 1회 스캔, 이후 O(1) 증분).
                    # 🚨 클램프 결함 수정(사용자 지적, 구현 전): running_extreme을 그대로
                    # extreme_price에 쓰면 p1 확정 직후 급격한 갭다운/갭업 시 파동 방향이
                    # 뒤집히는 구조 파괴가 생김 — 구코드의 max(p1["price"], memo_max_h,
                    # forming_high)/min(...) 암묵 클램프를 사용 시점에 명시적으로 재현
                    # (extreme_price = max/min(p1["price"], running_extreme_price)).
                    # 피봇 SET 자체가 5분봉 버전과 미세하게 달라질 수 있어(1분 wick 가시성)
                    # 이번엔 필드별 비트 동일 검증이 아니라 피봇 개수/가격/시각 비교 리포트로
                    # 검증(2021-09-07 BTC 플래시 크래시에서 5분봉이 놓친 진짜 저점을 1분봉이
                    # 잡아낸 사례 확인 — 결함 아닌 1분 정밀도의 정당한 이점). `_5m.csv` 로딩
                    # 자체를 제거(load_symbol이 df_1m만 반환) — 5m 데이터 제거 프로젝트
                    # (A/B/C/이 단계) 완료.
                    #
                    # v9d3 (2026-08-04): adx_5m/atr_5m/volatility_ratio/relative_volume_strength를
                    # df_5m(5분봉 DataFrame) 계산 대신 *_1m.csv의 5m_adx/5m_atr/5m_vol_ratio/
                    # 5m_vol_str 컬럼(전부 롤링/시프트 기반 인과적, 매분 갱신 — 직접 재구성해
                    # 상관계수 0.98~1.0으로 검증)으로 재계산. adx_5m/atr_5m/volatility_ratio는
                    # "지금"(i) 그대로 라이브 조회, relative_volume_strength는 a_p1_idx1m
                    # (B단계, 형성 중이면 라이브)에 인덱싱 — 기존 vol_idx=min(a_p1_idx,
                    # prev_5m_idx) 클램프 불필요(1분 배열이라 항상 인과적). calculate_adx/
                    # rolling_volume_strength(prep_features.py) 삭제(미사용).
                    #
                    # v9d2 (2026-08-04): div_rsi_gap/dist_from_div_peak_to_end/rsi_now를 5분봉
                    # RSI(df_5m["rsi"]) 대신 *_1m.csv의 5m_rsi 컬럼(1분 종가 기준 period-70
                    # RSI, 인과적 — Wilder RSI 재계산으로 검증됨)으로 재계산. 탐색 구간
                    # [a_p2, a_p1]을 rsi_1m_arr에서 직접 슬라이스해 argmax/argmin(기존
                    # calc_rsi_divergence와 동일 정의) — 끝점(a_p1)은 wave_duration_day와
                    # 똑같이 "형성 중이면 라이브, 고정되면 a_p1의 정밀 시각"으로 판정해
                    # div_price_gap(end_price=a_p1["price"])과 항상 같은 시점을 봄.
                    # a_p1/a_p2의 인덱스 표현(5분봉)은 안 바꿔서 피봇 검출/vol_idx/memo_key/
                    # ADX·ATR·volatility_ratio는 전부 무관·무변경. rsis_5m 의존 제거,
                    # calc_rsi_divergence(algorithm.py) 삭제(미사용).
                    #
                    # v9d1 (2026-08-04): wave_duration_day를 5분봉 인덱스 개수차×5분 대신
                    # 실제 극값 발생 시각(타임스탬프) 차로 재계산 — 파동이 형성 중인
                    # 5분봉 안에서도 매분 정확히 갱신됨(기존엔 5분에 한 번씩만 바뀜).
                    # a_p1/a_p2의 인덱스 표현(5분봉)은 안 바꿔서 RSI 다이버전스/
                    # relative_volume_strength/피봇 검출은 전부 무관·무변경. v9c와
                    # wave_duration_day 값만 다르고 나머지 필드는 bit-identical.
                    #
                    # v9c (2026-08-03): RSI 다이버전스 "last wave 외부 참조" 결함 수정
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
    ("i_5m", "i4"),                          # 2026-08-05부터 i_1m과 동일값(피봇이 1분 네이티브가 되며 중복) — 스키마 안정성 위해 필드는 유지
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
    ("relative_volume_strength", "f4"),      # 5m_vol_str(1분 컬럼), a_p1 피봇 위치 기준(형성 중이면 라이브)
    ("volatility_ratio", "f4"),              # 5m_vol_ratio(1분 컬럼), 2026-08-04부터 매분 라이브
    ("adx_5m", "f4"),                        # 5m_adx(1분 컬럼), 2026-08-04부터 매분 라이브
    ("atr_5m", "f4"),                        # 5m_atr(1분 컬럼), 2026-08-04부터 매분 라이브
    ("fib_pos", "f4"),                       # (close-end)/(start-end): 0=파동끝, 1=파동시작
    ("pre_hit_0382", "i1"),
    ("wave_age_min", "f4"),
    # --- 2026-07-20 타이밍 피처 (승률 천장 ~27%의 원인이 "전환 시점 판정 정보 부재"라는
    # 진단에 따른 추가 — 60d 런 칼손절 즉사 분석 참고). 전부 과거 데이터만 사용(인과적). ---
    ("rsi_now", "f4"),                       # 5m_rsi(1분 컬럼) 현재값 ("지금 과매수/과매도인가", 2026-08-04부터 매분 라이브)
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


def load_symbol(symbol, recent_days=None):
    # 2026-08-05: 5m 데이터 완전 제거 — 피봇 검출도 1분봉에서 직접 하므로 _5m.csv 자체를
    # 안 읽는다(prep_features.py 전체에서 5분봉 데이터 의존 소멸).
    df_1m = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_1m.csv"))
    if df_1m["ts"].dtype in (np.int64, np.float64):
        df_1m["ts_dt"] = pd.to_datetime(df_1m["ts"], unit="ms")
    else:
        df_1m["ts_dt"] = pd.to_datetime(df_1m["ts"])
    # OHLC 결측 행 제거 (BTC 1m 원본에 NaN 행 존재 확인됨 — NaN이 관측/보상에 전파되는 것 방지)
    df_1m = df_1m.dropna(subset=["open", "high", "low", "close"])
    df_1m = df_1m.sort_values("ts_dt").reset_index(drop=True)

    if recent_days is not None:
        cutoff = df_1m["ts_dt"].iloc[-1] - pd.Timedelta(days=recent_days)
        df_1m = df_1m[df_1m["ts_dt"] >= cutoff].reset_index(drop=True)
    return df_1m


def build_cache(symbol, recent_days=None, n=None, min_diff=None):
    params = PIVOT_PARAMS.get(symbol, {"n": PIVOT_N, "min_diff": PIVOT_MIN_DIFF})
    n = params["n"] if n is None else n
    min_diff = params["min_diff"] if min_diff is None else min_diff
    print(f"[{symbol}] pivot params: n={n} (1분봉 기준, PIVOT_PARAMS에 이미 x5 반영됨), min_diff={min_diff}")
    t0 = time.time()
    print(f"\n=== [{symbol}] loading data ===")
    df_1m = load_symbol(symbol, recent_days)
    total_1m = len(df_1m)
    print(f"1m rows: {total_1m:,}")

    # adx_5m/atr_5m/volatility_ratio/relative_volume_strength(2026-08-04 C단계):
    # *_1m.csv의 5m_adx/5m_atr/5m_vol_ratio/5m_vol_str 컬럼(전부 롤링/시프트 기반 인과적
    # 구성, 매분 갱신) 사용 — 5분봉 DataFrame(df_5m) 기반 계산 완전 제거.
    adx_1m_arr = df_1m["5m_adx"].fillna(0.0).values
    atr_1m_arr = df_1m["5m_atr"].fillna(0.0).values
    vol_ratio_1m_arr = df_1m["5m_vol_ratio"].fillna(1.0).values   # 기존 폴백값(1.0) 유지
    vol_str_1m_arr = df_1m["5m_vol_str"].fillna(0.0).values

    highs_1m = df_1m["high"].values
    lows_1m = df_1m["low"].values
    closes_1m = df_1m["close"].values
    # 🚨 반드시 ns 해상도를 경유해 epoch ms로 변환할 것 (2026-07-19 버그 수정).
    # pandas 3.x는 날짜 문자열을 datetime64[us]로 파싱하므로 astype(int64)의 단위가
    # 환경(pandas 버전)에 따라 ns/us로 달라짐 — us//10^6은 "초"가 되어 ts_1m 캐시가
    # 초 단위로 저장됐고, ms를 가정하는 wave_age_min(→ d5/d15 관측 게이트 전멸),
    # trades_per_month(1000배), yearly_breakdown(1970년) 등이 연쇄로 깨졌었음.
    ts_1m_ms = (df_1m["ts_dt"].astype("datetime64[ns]").astype("int64") // 10**6).values

    # RSI 다이버전스 3종(2026-08-04 B단계)용: 5분봉 RSI(rsis_5m) 대신 1분 컬럼 5m_rsi를
    # 직접 사용 — period-70 RSI(1분 종가 기준, 5분×14=70과 동일 스케일), Wilder RSI
    # 재계산으로 인과성(미래 데이터 미참조) 검증됨.
    rsi_1m_arr = df_1m["5m_rsi"].fillna(50.0).values

    # --- 전역 피봇 (2026-08-05: 5m 데이터 완전 제거 — 1분봉에서 직접 검출) ---
    # find_dynamic_pivots는 캔들 단위와 무관한 범용 함수 — df_1m을 그대로 넘기고, n도
    # PIVOT_PARAMS에 이미 1분봉 기준(x5)으로 저장돼 있어 그대로 씀(min_diff는 가격
    # 비율이라 스케일 불변, 그대로). 정순 1회 순회로 검출하고 각 피봇의 confirm_index(확정 봉)를
    # 함께 받는다 — confirm_index 이하만 쓰면 인과성 보장(algorithm.py 참고).
    print("Calculating global pivots on 1m...")
    global_pivots = find_dynamic_pivots(df_1m, n=n, min_diff=min_diff)
    print(f"pivots: {len(global_pivots)}  ({time.time() - t0:.1f}s)")

    # --- 1m 순회: 파동 매핑 + 피처 추출 ---
    print("Mapping 1m rows to waves and extracting features...")
    pivot_ptr = 0

    # pre-hit / wave-age 추적 (simulate_numba :114-131과 동일 리셋 규칙)
    prev_start = -1.0
    prev_end = -1.0
    wave_filter_hit = False
    prev_wave_start = -1.0
    wave_born_ms = -1

    # 확정 피봇(p1) 이후 "다음 반대방향 극점" 러닝 추적(2026-08-05, 5m 제거 최종단계):
    # 구버전은 "형성 중인 5분봉의 극값"(forming_high/low)과 "이미 마감된 5분봉들의
    # 극값"(memo_max_h/min_l)을 따로 추적해야 했는데, 1분봉은 모든 봉이 처리 즉시
    # "마감"이라 그 구분이 없어진다 — p1이 바뀔 때만 리셋되는 단일 러닝 익스트림 하나로
    # 통합. 리셋 시 p1_idx+1..현재 구간을 1회 스캔해 시드(확정 지연 구간까지 포함 —
    # 러닝 방식으로 "p1 인지 시점"부터만 추적하면 그 구간을 놓친다, B단계에서 이미 검증된
    # 결함 패턴), 이후 매 행 O(1) 증분. 총 스캔량은 각 행이 최대 한 번만 스캔되는
    # 텔레스코핑 합이라 전체 O(total_1m).
    running_extreme_price = 0.0
    running_extreme_idx = -1
    running_key_p1_idx = -1

    # RSI 다이버전스 탐색 구간 메모(2026-08-04 B단계): (a_p2_idx1m, a_p1_idx1m)가 탐색
    # 구간을 완전히 결정하므로 이 조합이 바뀔 때만 argmax/argmin을 다시 돌린다.
    a_p2_idx1m = -1
    div_win_key = None
    div_rsi_ref = 0.0
    div_ref_price = 0.0
    j_idx1m = 0

    rows = []
    t_loop = time.time()

    for i in range(total_1m):
        if i % 200000 == 0 and i > 0:
            print(f"  {i:,}/{total_1m:,}  ({time.time() - t_loop:.1f}s)")

        if i < WARMUP_1M:
            continue

        # 버그② 수정 (admission 가드): 구버전은 `index <= idx_5m - n`으로 거리 조건만 보고
        # 가격 조건(min_diff)을 빼먹어, 반등이 min_diff에 도달하기 전에 피봇을 확정 처리했다
        # (실측 최대 47봉=3.9시간 미래 선행). 검출기가 두 조건을 모두 만족한 봉을
        # confirm_index로 알려주므로, 그 봉이 마감된 뒤에만 사용한다.
        while pivot_ptr < len(global_pivots) and global_pivots[pivot_ptr]["confirm_index"] < i:
            pivot_ptr += 1
        if pivot_ptr < 2:
            continue

        last_pivots_desc = global_pivots[max(0, pivot_ptr - 10):pivot_ptr][::-1]  # 최신순
        p1 = last_pivots_desc[0]
        p1_idx = p1["index"]

        # 확정 피봇 이후의 신규 극점(forming 포함) 추적 — V7/V8과 동일
        extreme_type = "high" if p1["type"] == "low" else "low"
        if running_key_p1_idx != p1_idx:
            # p1이 막 바뀐 첫 행 — p1_idx+1..i 구간을 1회 스캔해 정확히 시드
            seg_h = highs_1m[p1_idx + 1: i + 1]
            seg_l = lows_1m[p1_idx + 1: i + 1]
            if extreme_type == "high":
                off = int(seg_h.argmax())
                running_extreme_price, running_extreme_idx = seg_h[off], p1_idx + 1 + off
            else:
                off = int(seg_l.argmin())
                running_extreme_price, running_extreme_idx = seg_l[off], p1_idx + 1 + off
            running_key_p1_idx = p1_idx
        else:
            if extreme_type == "high":
                if highs_1m[i] > running_extreme_price:
                    running_extreme_price, running_extreme_idx = highs_1m[i], i
            else:
                if lows_1m[i] < running_extreme_price:
                    running_extreme_price, running_extreme_idx = lows_1m[i], i

        # 🚨 클램프 필수(2026-08-05 결함 수정, 구현 전 사용자 지적): running_extreme을
        # 그대로 쓰면 p1 확정 직후 급격한 갭다운/갭업으로 이후 구간의 실제 극값이
        # p1["price"]를 역행할 때(예: low=100 확정 직후 그 뒤 모든 high가 100 미만),
        # "low(100) 다음에 그보다 낮은 high(90)"처럼 파동 방향이 뒤집히는 구조 파괴가
        # 생긴다. 구코드의 max(p1["price"], memo_max_h, forming_high)/min(...) 암묵
        # 클램프를 사용 시점에 명시적으로 재현.
        if extreme_type == "high":
            extreme_price = max(p1["price"], running_extreme_price)
            extreme_idx = running_extreme_idx if running_extreme_price > p1["price"] else p1_idx
        else:
            extreme_price = min(p1["price"], running_extreme_price)
            extreme_idx = running_extreme_idx if running_extreme_price < p1["price"] else p1_idx

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

        # wave_duration_day (2026-08-05: 피봇이 1분봉 네이티브가 되며 a_p1["index"]/
        # a_p2["index"]가 이미 1분 행 번호라 바로 조회 — A/B단계의 "5분 인덱스 ↔ 1분
        # 정밀 시각" 변환 장치(ts_5m_*_precise_ms 등)가 통째로 불필요해짐).
        a_p1_ts_ms = ts_1m_ms[a_p1["index"]]
        a_p2_ts_ms = ts_1m_ms[a_p2["index"]]
        wave_duration_day = max(0.001, (a_p1_ts_ms - a_p2_ts_ms) / 86_400_000)

        # wave age + RSI 다이버전스 탐색 시작점(B단계 로직 유지, 2026-08-05: a_p2_idx1m을
        # a_p2["index"] 그대로 대입 — 이미 1분 인덱스라 변환 불필요). 파동 시작점
        # (start_price)이 바뀌면 신규 파동으로 간주.
        if start_price != prev_wave_start:
            prev_wave_start = start_price
            wave_born_ms = ts_1m_ms[i]
            a_p2_idx1m = a_p2["index"]
            div_win_key = None  # 아래에서 강제 재계산되도록
        wave_age_min = (ts_1m_ms[i] - wave_born_ms) / 60000.0

        # RSI 다이버전스 3종(B단계 로직 유지, 2026-08-05: a_p1_idx1m도 a_p1["index"]
        # 그대로 대입 — "형성중/고정" 분기 자체가 불필요해짐, 더 이상 형성중 5분봉이
        # 없으므로). div_price_gap(end_price=a_p1["price"])과 항상 같은 시점을 보게 되어
        # 세 필드 간 시점 분열이 없다.
        a_p1_idx1m = a_p1["index"]

        div_win_key_now = (a_p2_idx1m, a_p1_idx1m)
        if div_win_key_now != div_win_key:
            div_win_key = div_win_key_now
            window = rsi_1m_arr[a_p2_idx1m: a_p1_idx1m + 1]
            j_off = int(np.argmax(window) if is_bullish else np.argmin(window))
            j_idx1m = a_p2_idx1m + j_off
            div_rsi_ref = float(window[j_off])
            div_ref_price = float(highs_1m[j_idx1m] if is_bullish else lows_1m[j_idx1m])

        rsi_now = rsi_1m_arr[i]              # 관측값: 항상 라이브(파동과 무관)
        rsi_end = rsi_1m_arr[a_p1_idx1m]     # 다이버전스 계산용: a_p1 위치에 고정
        div_rsi_gap = max(0.0, (div_rsi_ref - rsi_end) if is_bullish else (rsi_end - div_rsi_ref))
        dist_from_div_peak_to_end = max(0.001, (a_p1_ts_ms - ts_1m_ms[j_idx1m]) / 86_400_000)
        div_price_gap = (abs(end_price - div_ref_price) / div_ref_price * 100
                         if div_ref_price > 0 else 0.0)

        # 볼륨 상대강도(C단계 로직 유지): 파동 끝 피봇 캔들(a_p1) 볼륨의 트레일링
        # 500(5분봉 환산)캔들 백분위. a_p1_idx1m에 그대로 인덱싱.
        vol_strength = vol_str_1m_arr[a_p1_idx1m]

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

        denom = start_price - end_price
        fib_pos = (closes_1m[i] - end_price) / denom if denom != 0 else 0.0

        # 타이밍 피처: 트레일링 수익률 (1m 행 인덱스 기준 — 데이터 갭 시 근사, 드묾)
        ret_5m = closes_1m[i] / closes_1m[i - 5] - 1.0 if i >= 5 else 0.0
        ret_15m = closes_1m[i] / closes_1m[i - 15] - 1.0 if i >= 15 else 0.0
        ret_1h = closes_1m[i] / closes_1m[i - 60] - 1.0 if i >= 60 else 0.0

        rows.append((
            i, i, int(ts_1m_ms[i]),   # i_5m: 2026-08-05부터 i_1m과 동일값(피봇이 1분 네이티브가 되며 중복)
            highs_1m[i], lows_1m[i], closes_1m[i],
            1 if is_bullish else 0,
            start_price, end_price,
            wave_scale, wave_duration_day,
            div_rsi_gap, dist_from_div_peak_to_end, div_price_gap,
            vol_strength, vol_ratio_1m_arr[i],
            adx_1m_arr[i], atr_1m_arr[i],
            fib_pos, 1 if wave_filter_hit else 0,
            wave_age_min,
            rsi_now,   # 5m_rsi(1분 컬럼) 현재값 — 2026-08-04부터 매분 라이브
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
