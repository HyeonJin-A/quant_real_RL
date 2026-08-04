import pandas as pd
import numpy as np

def find_dynamic_pivots(df, n=10, min_diff=0.05):
    """시간 오름차순 캔들에서 피봇을 인과적으로 검출한다. (2026-08-03 재작성 — 피봇 admission 룩어헤드 버그 수정)

    피봇 확정 조건은 두 개의 AND — 둘 다 그 시점 데이터만으로 판정 가능하다:
      (a) 거리:  t − P >= n        (P로부터 최소 n봉 경과)
      (b) 가격:  가격이 P로부터 min_diff 이상 반대 방향으로 이동

    같은 방향으로 더 극단이 나오면 후보를 즉시 그 봉으로 갱신한다(지연 없음).

    🚨 구버전의 결함: 확정 조건이 "t−P >= n"이 아니라 "봉 i가 앞뒤 n봉 중 극점"(N-Bar)이었고
    그 판정이 `iloc[i+1:i+n+1]`로 미래 n봉을 참조했다. 게다가 `prep_features`가 이 함수를
    시간 역순 df에 적용한 뒤 인덱스를 되돌리는 방식이었는데, 대칭 N-Bar 판정을 역순에 걸면
    원래 시간축 기준으로는 오히려 미래 확정을 더 크게 앞당기는 결과가 됐다. 그리고 그 바깥의
    admission 가드(`prep_features`)가 (a) 거리 조건만 보고 (b) 가격 조건을 빼먹어, 반등이
    min_diff에 도달하기도 전에 피봇을 확정 처리했다 — 반등이 늦게 오는 파동일수록 미래를
    크게 앞당겨 썼다(실측 최대 47봉=3.9시간). 이번 재작성은 정순 1회 순회로 검출해
    두 조건을 모두 만족한 봉을 즉시 confirm_index로 기록하는 구조로 이 결함을 근본 해결한다.

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


