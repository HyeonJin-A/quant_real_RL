import pandas as pd
import numpy as np

def find_dynamic_pivots(df, n=10, min_diff=0.05):
    """시간 오름차순 캔들에서 피봇을 인과적으로 검출한다. (2026-08-01 전면 재작성)

    피봇 확정 조건은 두 개의 AND — 둘 다 그 시점 데이터만으로 판정 가능하다:
      (a) 거리:  t − P >= n        (P로부터 최소 n봉 경과)
      (b) 가격:  가격이 P로부터 min_diff 이상 반대 방향으로 이동

    같은 방향으로 더 극단이 나오면 후보를 즉시 그 봉으로 갱신한다(지연 없음).
    "후보"와 "확정"은 별개 개념이 아니라, 매 시점 하나의 last wave가 존재하고
    위 두 조건이 충족되는 순간 직전 극점이 피봇으로 굳는 것이다.

    🚨 구버전(~2026-07-31)의 두 가지 결함을 수정한 것:
      1) 조건 (a)가 "t−P >= n"이 아니라 "봉 i가 앞뒤 n봉 중 극점"(N-Bar)이었고,
         그 판정이 `iloc[i+1:i+n+1]`로 **미래 n봉을 참조**했다.
      2) `prep_features`가 이 함수를 **시간 역순 df**에 적용했다 — 스캔이 최신 봉에서
         과거로 진행되는 구조라 라이브에서 재현 불가능했다.
      그리고 admission 가드가 (a)만 보고 (b)를 빼먹어, 반등이 min_diff에 도달하기도
      전에 피봇을 확정 처리했다 — 반등이 늦게 오는 파동일수록 미래를 크게 앞당겨 썼다.

    :param df: 시간 오름차순 캔들 (필수 컬럼 high/low, 선택 ts)
    :param n: 피봇 간 최소 캔들 간격
    :param min_diff: 방향 전환으로 인정할 가격 비율
    :return: [{'index', 'type', 'price', 'confirm_index', 'time'}] — 확정된 피봇만.
        confirm_index = 이 피봇이 확정된 봉. **그 봉까지의 데이터만으로 판정 가능**하므로
        호출부는 `confirm_index <= 현재봉`인 피봇만 사용하면 인과성이 보장된다.
        아직 확정되지 않은 마지막 후보는 포함하지 않는다(호출부가 forming 극점으로 별도 추적).
    """
    highs = df['high'].to_numpy(dtype=float)
    lows = df['low'].to_numpy(dtype=float)
    idx = df.index.to_numpy()
    ts = df['ts'].to_numpy() if 'ts' in df.columns else None

    pivots = []
    if len(df) == 0:
        return pivots

    curr_type = 'high'          # 초기 방향 (첫 전환에서 실제 구조에 맞춰 정렬됨)
    curr_i = 0                  # 후보의 위치 인덱스
    curr_price = highs[0]

    for i in range(1, len(df)):
        if curr_type == 'high':
            if highs[i] > curr_price:
                curr_i, curr_price = i, highs[i]          # 후보 갱신 (지연 없음)
            elif lows[i] < curr_price * (1 - min_diff) and (i - curr_i) >= n:
                pivots.append({
                    'index': int(idx[curr_i]), 'type': 'high', 'price': float(curr_price),
                    'confirm_index': int(idx[i]),
                    'time': str(ts[curr_i]) if ts is not None else None,
                })
                curr_type = 'low'
                curr_i, curr_price = i, lows[i]
        else:
            if lows[i] < curr_price:
                curr_i, curr_price = i, lows[i]
            elif highs[i] > curr_price * (1 + min_diff) and (i - curr_i) >= n:
                pivots.append({
                    'index': int(idx[curr_i]), 'type': 'low', 'price': float(curr_price),
                    'confirm_index': int(idx[i]),
                    'time': str(ts[curr_i]) if ts is not None else None,
                })
                curr_type = 'high'
                curr_i, curr_price = i, highs[i]

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
