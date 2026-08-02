"""
파동 감지 엔진. find_dynamic_pivots만 유지.

2026-08-02 죽은 코드 정리 (RSI-div-개편이전-핵심결함.md 수정안 4):
get_pivots_with_extreme_candle / calc_fibo_levels / check_rsi_divergence /
get_relative_volume_strength / normalize / calc_rule_based_score 삭제 —
이 레포 내 호출자 없음(실거래 레포 비공유 확인). 극점 봉 인덱스 추적(idxmax 패턴)은
prep_features.py의 memo argmax 로직으로, 볼륨 상대강도는 rolling_volume_strength로 계승됨.
"""


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
