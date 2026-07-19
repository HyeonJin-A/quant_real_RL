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


def get_pivots_with_extreme_candle(df, pivots, min_diff=0.03):
    """
    현재 시점에서 가장 최근의 극점(extreme_candle)을 찾고 pivot 목록에 추가하여 반환함
    :param df: 시간 역순으로 정렬된 캔들 데이터
    :param pivots: 시간 역순으로 정렬된 피봇 목록
    :param min_diff: 최신피봇과 신규극점의 최소 차이
    """
    if not pivots or len(pivots) < 2:
        return pivots

    # 피봇이 너무 최신이면 극점 업데이트 안함
    if pivots[0]['index'] < 4:
        return pivots

    # 데이터 슬라이싱 후 극값 추적 (최신 피봇 자체는 제외)
    last_pivot = pivots[0]
    pivot_idx = last_pivot['index']
    post_pivot_df = df.loc[:pivot_idx - 1]

    if post_pivot_df.empty:
        return pivots

    extreme_type = 'high' if (last_pivot['type'] == 'low') else 'low'
    extreme_idx = (post_pivot_df['high'].idxmax() if extreme_type == 'high'
                   else post_pivot_df['low'].idxmin())

    last_pivot_price = pivots[0]['price']
    extreme_candle_price = float(df[extreme_type].loc[extreme_idx])

    # 최신피봇과 신규극점의 가격 차이가 min_diff 미만이면 극점 업데이트 안함
    if abs(last_pivot_price - extreme_candle_price) < min(last_pivot_price, extreme_candle_price) * min_diff:
        return pivots

    extreme_candle = {
        'index': extreme_idx,
        'time': df['ts'].loc[extreme_idx],
        'type': extreme_type,
        'price': float(df[extreme_type].loc[extreme_idx]),
    }
    return [extreme_candle] + pivots


def calc_fibo_levels(p1, p2):
    diff = abs(p1 - p2)
    # 피보나치 기본 비율
    ratios = [0.382, 0.5, 0.618]
    levels = {}

    if p1 < p2:
        # 상승 후 되돌림 (Retracement from High)
        # 고점에서 얼마나 내려오는지 계산
        for r in ratios:
            levels[str(r)] = round(p2 - (diff * r), 2)
    else:
        # 하락 후 반등 (Rebound from Low)
        # 저점에서 얼마나 올라오는지 계산
        for r in ratios:
            levels[str(r)] = round(p2 + (diff * r), 2)

    return levels


def check_rsi_divergence(df, pivots, min_candle_dist=5):
    """
    과거 후보 구간에서 비교 대상을 찾아 다이버전스 성립 여부를 반환함
    :param df: 시간 역순으로 정렬된 캔들 데이터
    :param pivots: 시간 역순으로 정렬된 피봇 목록
    :param min_candle_dist: 다이버전스가 성립되기 위한 최소 캔들 간격
    """
    pivot_idx = pivots[1]['index']
    extreme_idx = pivots[0]['index']
    extreme_type = pivots[0]['type']
    extreme_candle = df.loc[extreme_idx]

    # [2-1] 후보 구간 추출
    between_df = df.loc[extreme_idx + 1 + min_candle_dist: pivot_idx]
    if between_df.empty:
        return False, None

    # [2-2] 타입에 따른 설정값 세팅
    if extreme_type == 'high':
        prev_idxes = between_df['high'].nlargest(10).index
        sort_col, agg_func, div_type = 'high', 'idxmax', 'bearish'
        rsi_threshold = lambda x: x > 70
        rsi_comp = lambda a, b: a > b  # RSI가 낮아져야 함 (Bearish)
    else:
        prev_idxes = between_df['low'].nsmallest(10).index
        sort_col, agg_func, div_type = 'low', 'idxmin', 'bullish'
        rsi_threshold = lambda x: x < 30
        rsi_comp = lambda a, b: a < b  # RSI가 높아져야 함 (Bullish)

    candidates = df.loc[prev_idxes].sort_values(by='ts', ascending=False)

    # [2-3] 비교 대상(prev_extreme_candles) 선별
    prev1_idx = getattr(candidates[sort_col], agg_func)()
    prev_extreme_candles = [candidates.loc[prev1_idx]]

    prev2_candidates = candidates[candidates.index >= prev1_idx + min_candle_dist]
    if not prev2_candidates.empty:
        prev2_idx = getattr(prev2_candidates[sort_col], agg_func)()
        prev_extreme_candles.append(prev2_candidates.loc[prev2_idx])

    # [3] 최종 다이버전스 체크
    for prev_candle in prev_extreme_candles:
        if rsi_threshold(prev_candle['rsi']) and rsi_comp(prev_candle['rsi'], extreme_candle['rsi']):
            return True, {
                'type': div_type,
                'candle1': prev_candle,
                'candle2': extreme_candle,
            }
    return False, None


def get_relative_volume_strength(df, target_idx, period=500):
    """
    특정 캔들의 거래량 상대강도를 계산합니다.

    [입력]
    - df: 캔들 데이터프레임 (필수 컬럼: 'volume')
    - target_idx: 계산할 캔들의 인덱스
    - period: 이동평균을 구할 기간 (기본값 20)

    [출력]
    - 상대강도 백분위
    """
    # 타겟 캔들까지의 거래량 슬라이싱 (period 개수만큼)
    # iloc를 사용하여 인덱스 기준으로 정확히 추출
    volume_window = df['volume'].iloc[target_idx: target_idx + period]

    min_vol = volume_window.min()
    max_vol = volume_window.max()
    current_vol = df['volume'].iloc[target_idx]

    # 0으로 나누는 경우 방지
    if max_vol == min_vol:
        return 0.0

    if min_vol == 0:
        min_vol = 1

    # Min-Max 공식: (x - min) / (max - min)
    strength = (current_vol - min_vol) / (max_vol - min_vol)

    # 1.0을 초과하거나 0.0 미만일 수 있으므로 클리핑(범위 제한)
    return round(max(0.0, min(1.0, float(strength))), 3)


def normalize(value, min_val, max_val, is_positive=True, max_score=10.0):
    """
    모든 스코어 항목을 0 ~ max_score 사이로 정규화하는 함수
    :param value: 현재 측정값
    :param min_val: 스코어 0이 되는 기준점 (하한선)
    :param max_val: 최고 점수(max_score)가 되는 기준점 (상한선)
    :param is_positive: True면 높은 값이 더 높은 점수
    :param max_score: 해당 항목의 만점 (가중치 배점)
    """
    # 1. 이상치 방어 (Clipping)
    # value가 min_val보다 작으면 min_val로, max_val보다 크면 max_val로 고정합니다.
    # 즉, 10% 이상이 들어와도 10%로 처리되어 무조건 만점을 받게 됩니다.
    clipped_value = np.clip(value, min_val, max_val)

    # 2. 0.0 ~ 1.0 사이로 기본 정규화
    if max_val == min_val:  # 0으로 나누기 방지
        return 0.0
    score = (clipped_value - min_val) / (max_val - min_val)

    # 3. 역방향 처리 (낮을수록 좋은 지표)
    if not is_positive:
        score = 1.0 - score

    # 4. 배점(max_score)을 곱해주고 소수점 둘째 자리까지 반올림
    return round(score * max_score, 2)


def calc_rule_based_score(last_wave, interval="15m", custom_scoring=None, custom_weights=None):
    from Const import INTERVAL_SETTINGS

    settings = INTERVAL_SETTINGS.get(interval, INTERVAL_SETTINGS["15m"])
    sc = custom_scoring if custom_scoring else settings.get("scoring", {
        "wave_scale_min": 1.0,
        "wave_scale_max": 5.0,
        "wave_duration_min": 0.1,
        "wave_duration_max": 3.0,
        "div_price_gap_max": 3.0,
        "vol_strength_max": 0.5,
        "vol_ratio_min": 1.5,
        "vol_ratio_max": 3.0,
        "adx_min": 15.0,
        "adx_max": 35.0
    })
    
    w = custom_weights if custom_weights else settings.get("weights", {
        "wave_scale": 15.0,
        "wave_duration": 10.0,
        "rsi_base": 15.0,
        "rsi_str": 10.0,
        "rsi_gap": 10.0,
        "vol_str": 10.0,
        "vol_ratio": 15.0,
        "adx": 15.0
    })

    score = 0

    # 1. wave 형태 (Wave Scale & Duration)
    score += normalize(last_wave['wave_scale_percent'], sc['wave_scale_min'], sc['wave_scale_max'], is_positive=True, max_score=w['wave_scale'])
    score += normalize(last_wave['wave_duration_day'], sc['wave_duration_min'], sc['wave_duration_max'], is_positive=True, max_score=w['wave_duration'])

    # 2. RSI 다이버전스 (RSI Divergence)
    rsi_info = last_wave['rsi']
    if rsi_info['rsi_divergence']:
        score += w['rsi_base']

        # RSI 스케일 0~100 대응을 위한 역방향 처리
        rsi = rsi_info['rsi_previous'] if last_wave['wave_direction_is_bullish'] else 100 - rsi_info['rsi_previous']
        rsi_score_val = rsi - 70
        score += normalize(rsi_score_val, 0, 10, is_positive=True, max_score=w['rsi_str'])

        score += normalize(rsi_info['divergence_price_gap_percent'], 0, sc['div_price_gap_max'], is_positive=False, max_score=w['rsi_gap'])

    # 3. 거래량 상대강도 (Volume Strength)
    score += normalize(last_wave['relative_volume_strength_of_the_latest_pivot'], 0.0, sc.get('vol_strength_max', 1.0), is_positive=True, max_score=w['vol_str'])
    
    # 4. 변동성 폭발 (Volatility Ratio)
    vol_ratio = last_wave.get('volatility_ratio', 1.0)
    score += normalize(vol_ratio, sc.get('vol_ratio_min', 1.5), sc.get('vol_ratio_max', 2.5), is_positive=True, max_score=w['vol_ratio'])

    # 5. ADX 추세 강도
    adx_val = last_wave.get('adx', 0.0)
    score += normalize(adx_val, sc.get('adx_min', 15.0), sc.get('adx_max', 35.0), is_positive=True, max_score=w['adx'])

    return round(float(score), 2)