import pandas as pd
import numpy as np

def find_dynamic_pivots(df, n=10, min_diff=0.05):
    """
    :param df: 시간순으로 정렬된 캔들 데이터
    :param n: N-Bar 알고리즘변수 (피봇 간격이 최소 N개 캔들)
    :param min_diff: 피봇 방향 전환 조건이 되는 가격 비율
    :return:
    """
    pivots = []
    curr_type = 'high'  # 초기 방향 설정 (상황에 따라 low로 시작 가능)
    first_idx = int(df.index[0])
    curr_pivot = {'index': first_idx, 'time': str(df['ts'].iloc[0]), 'type': 'high', 'price': df['high'].iloc[0]}

    for i in range(1, len(df)):
        index_i = int(df.iloc[i].name)
        high_i = float(df['high'].iloc[i])
        low_i = float(df['low'].iloc[i])
        ts_i = str(df['ts'].iloc[i])

        if curr_type == 'high':
            # 1. 단순 가격 갱신: N-Bar 상관없이 더 높으면 무조건 업데이트
            if high_i > curr_pivot['price']:
                curr_pivot = {'index': index_i, 'time': ts_i, 'type': 'high', 'price': high_i}

            # 2. 전환 조건: 충분히 하락했고 + N-Bar 저점 조건 만족 시
            elif (low_i < curr_pivot['price'] * (1 - min_diff)):
                is_low_n_bar = all(low_i < df['low'].iloc[i - n:i]) and \
                               all(low_i < df['low'].iloc[i + 1:i + n + 1])
                if is_low_n_bar:
                    pivots.append(curr_pivot)  # 고점 확정
                    curr_type = 'low'
                    curr_pivot = {'index': index_i, 'time': ts_i, 'type': 'low', 'price': low_i}

        elif curr_type == 'low':
            # 1. 단순 가격 갱신: 더 낮으면 무조건 업데이트
            if low_i < curr_pivot['price']:
                curr_pivot = {'index': index_i, 'time': ts_i, 'type': 'low', 'price': low_i}

            # 2. 전환 조건: 충분히 상승했고 + N-Bar 고점 조건 만족 시
            elif (high_i > curr_pivot['price'] * (1 + min_diff)):
                is_high_n_bar = all(high_i > df['high'].iloc[i - n:i]) and \
                                all(high_i > df['high'].iloc[i + 1:i + n + 1])
                if is_high_n_bar:
                    pivots.append(curr_pivot)  # 저점 확정
                    curr_type = 'high'
                    curr_pivot = {'index': index_i, 'time': ts_i, 'type': 'high', 'price': high_i}

    pivots.append(curr_pivot)
    return pivots


def calc_rsi_divergence(rsis, highs, lows, start_idx, end_idx, is_bullish):
    """
    파동(leg) 내부의 "모멘텀 천장 대비 현재 모멘텀의 괴리"를 연속값으로 계산한다. (2026-07-31 재설계)

    고전적인 2-피크 다이버전스(서로 다른 스윙 고점끼리 비교)가 아니다. 파동 길이 중앙값이
    14봉(BTC)/26봉(ETH)에 불과해 **파동 내부에 되돌림으로 분리된 두 개의 스윙 고점이
    존재하지 않는 경우가 대부분**이라, 피크 탐색 자체를 포기하고 argmax/argmin으로 재정의했다.

    파동 끝 극점(a_p1)의 가격은 정의상 파동 내 가격 극값이므로 "higher high" 조건은 항상
    자동 충족된다 → 신호는 순수하게 RSI 비교로 환원된다. 그래서 가격 조건을 따로 보지 않는다.

    구버전(피크 탐색)이 쓰던 임의 상수 — 후보 상위 10개(`nlargest(10)`), 최소 캔들 간격 5,
    RSI 임계값 70/30, prev1/prev2 OR 판정 — 을 **하나도 쓰지 않는다.** 특히 최소 캔들 간격 5는
    BTC 파동의 21.1%를 계산도 해보기 전에 탈락시키고 있었고(파동이 5봉 이하면 후보 구간이 공백),
    같은 상수가 ETH에선 8.5%만 탈락시켜 심볼 간 비대칭까지 만들고 있었다.

    :param rsis: 5m RSI 배열 (시간 오름차순)
    :param highs: 5m 고가 배열 (시간 오름차순)
    :param lows: 5m 저가 배열 (시간 오름차순)
    :param start_idx: 파동 시작 피봇 인덱스 (a_p2)
    :param end_idx: 파동 끝 기준 인덱스. RSI를 읽을 수 있는 마지막 인덱스여야 하므로
                    호출부에서 a_p1을 직전 마감봉으로 클램프해서 넘긴다 (인과성)
    :param is_bullish: 상승 파동(끝점이 고점)이면 True
    :return: (rsi_gap, dist_bars, ref_price)
        - rsi_gap: 파동 내 RSI 극값 − 파동 끝 RSI (방향 조정, 항상 >= 0).
                   0이면 모멘텀이 가격과 동시에 정점 = 확인(confirmation),
                   클수록 모멘텀이 먼저 꺾인 뒤 가격만 진행 = 다이버전스
        - dist_bars: RSI 극점에서 파동 끝까지의 5m 봉 개수 (>= 0). 0이면 방금 정점
        - ref_price: RSI 극점 봉의 가격(상승이면 고가, 하락이면 저가).
                     가격 갭은 파동 극점 가격이 1m마다 갱신되므로 호출부에서 계산한다
    """
    if end_idx <= start_idx:
        return 0.0, 0, 0.0

    window = rsis[start_idx:end_idx + 1]
    j = start_idx + int(np.argmax(window) if is_bullish else np.argmin(window))

    rsi_ref = float(rsis[j])
    rsi_end = float(rsis[end_idx])
    # rsi_ref는 end_idx를 포함한 구간의 극값이므로 방향 조정 후 항상 >= 0 (max는 부동소수 방어)
    rsi_gap = max(0.0, (rsi_ref - rsi_end) if is_bullish else (rsi_end - rsi_ref))
    ref_price = float(highs[j] if is_bullish else lows[j])

    return rsi_gap, end_idx - j, ref_price
