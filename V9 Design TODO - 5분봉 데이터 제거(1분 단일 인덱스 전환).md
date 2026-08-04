# V9 Design TODO — 5분봉 데이터 제거, 1분 단일 인덱스 아키텍처 전환

상태: **🎉 전 단계(A/B/C + 5m 데이터 완전 제거) 적용 완료 및 검증 통과 (2026-08-05,
`CACHE_VER=v9d4`). 이 문서의 목표(1분 단일 인덱스 아키텍처 전환) 달성 — `prep_features.py`가
`_1m.csv`만 읽음.**

**A 단계 적용 결과 요약**: `src/prep_features.py` 3곳 수정(설계 결함 재점검 후 확정된
버전, 3장 참고). 풀 캐시(BTC/ETH) 재생성 후 필드별 회귀 — `wave_duration_day` 딱 한
필드만 값이 다르고 나머지 전부 비트 단위 동일. 같은 파동 유지 구간 내 단조 비감소 검증:
ETH는 342만 행 전부 통과, BTC는 345만 행 중 1행(2020-05-13 16:35 5분봉)에서 위반 발견
→ 추적 결과 **이번 변경이 만든 결함이 아니라 원본 데이터 자체의 결함**(그 5분봉의 `low`
컬럼이 실제 1분봉 5개의 최저가와 불일치, 5m/1m CSV 간 원본 데이터 불일치, 발생빈도
1/345만) — 기존(미변경) `memo_min_l`이 그 오차를 그대로 드러낸 것뿐이라 이번 수정 범위
밖으로 판단, 기록만 남기고 조치 안 함. `distribution_report` p99(BTC 1.05일/ETH 1.81일)도
`NORM["duration_log_max"]` 커버리지 이내라 `env.py` 변경 불필요.

**B 단계 적용 결과 요약**: `div_rsi_gap`/`dist_from_div_peak_to_end`/`rsi_now`를 `*_1m.csv`의
`5m_rsi`(1분, period-70, Wilder RSI 재계산으로 인과성 검증됨) 기반으로 재설계. 최초
"O(1) 러닝 익스트림"안이 사용자 검토로 3가지 결함(확정 지연 구간 누락/고점 이후
덮어쓰기/세 필드 간 시점 분열)이 드러나 반려, "정적 배열 슬라이스+argmax, 탐색 끝은
`wave_duration_day`와 동일한 형성/고정 분기"로 재설계해 승인(4장 참고). 풀 캐시(BTC/ETH)
재생성 후 필드별 회귀(바뀐 필드 4개만, 나머지 비트 단위 동일), `a_p1` 정지 구간
전부(BTC 323만행/ETH 330만행)에서 세 필드 동시 동결 확인(위반 0건), 확정 지연 구간
누락 없음을 실제 파동 샘플로 확인. `distribution_report` p99(BTC 10.5pt/ETH 13.5pt)도
`NORM["div_rsi_gap_max"]=30.0` 커버리지 이내라 `env.py` 변경 불필요.

**C 단계 적용 결과 요약**: `adx_5m`/`atr_5m`/`volatility_ratio`/`relative_volume_strength`를
`*_1m.csv`의 `5m_adx`/`5m_atr`/`5m_vol_ratio`/`5m_vol_str`(1분 컬럼, 직접 재구성한 값과
상관계수 0.98~1.0으로 인과성 검증됨) 기반으로 재설계. 앞의 3개는 "현재 시장 레짐"이라
"지금"(`i`) 그대로 라이브 조회, `relative_volume_strength`만 성격이 달라(파동 끝 피봇
캔들 자체의 값) B단계의 `a_p1_idx1m`을 재사용해 인덱싱 — 기존 `vol_idx` 클램프는
불필요해져 제거(더 정확해짐). 풀 캐시(BTC/ETH) 재생성 후 필드별 회귀(바뀐 필드 4개만,
나머지 비트 단위 동일), 같은 5분봉 그룹 내에서도 80~98% 행이 값이 갱신됨 확인(기존엔
0%, 5분에 한 번만 바뀜), `relative_volume_strength` 범위 `[0,1]` 확인.
`distribution_report` p99(`volatility_ratio` 3.6~3.8, `atr_close_percent` 0.58~0.84)도
`NORM["vol_ratio_log_max"]`/`atr_pct_max` 커버리지 이내라 `env.py` 변경 불필요.
`design_spec.md` 반영 완료.

## 1. 배경

`prep_features.py`는 파동(`a_p1`/`a_p2`, 피봇)을 5분봉 인덱스로, 가격/거래량 등 나머지를
1분봉 인덱스로 갖는 이중 인덱스 구조다. 이 세션에서 겪은 `div_end_idx` stale-reference
버그(78% 행 오염)를 포함해, 두 인덱스를 짝지어 계산해야 하는 지점마다 반복적으로 결함이
발생해왔다.

기존 5분 갱신 주기 항목 3가지:
- **A**: `wave_duration_day`
- **B**: RSI 관련 — `div_rsi_gap`, `dist_from_div_peak_to_end`, `rsi_now`
- **C**: 그 외 — `volatility_ratio`, `relative_volume_strength`, `adx_5m`/`atr_5m`

## 2. 논의 경과 (핵심 교훈 — 왜 A가 다시 독립적인 단계가 됐는지)

여러 차례 순서를 재검토하는 과정에서 "A(`wave_duration_day`)를 5m 데이터 없이 계산하려면
피봇 검출 자체가 1분봉 네이티브여야 한다"고 결론 내렸었고, 그 결론을 따라가다 보니:
- 피봇 인덱스가 1분 스케일이 되면 그 인덱스로 5분봉 배열을 조회하는 RSI 다이버전스·
  `relative_volume_strength`도 같이 깨지므로 안 건드릴 수 없음 → A/B가 강제로 묶임
- "0단계(공통 선행 구조개편)"를 따로 빼자는 얘기까지 나옴
- 그런데 "0단계 이후에도 관측값이 하나도 안 바뀌어야 한다"는 기준을 적용해보니, **애초에
  a_p1 인덱스의 표현 방식(5분/1분)을 바꾸는 것 자체가 이미 "값이 바뀌는 변경"**이라
  value-neutral한 "0단계"라는 개념 자체가 성립 불가능하다는 게 드러남

**결정적 재검토**: `wave_scale_percent`는 이미 매분 갱신되고 있다 — `forming_high`/
`forming_low`가 애초에 `highs_1m[i]`/`lows_1m[i]`를 매 행 스캔해서 값을 계산하기
때문. `idx_5m`은 "몇 번째 5분봉이었는지" 레이블로만 쓰였을 뿐 값 계산 자체에는
관여하지 않았다. 즉 `wave_duration_day`가 안 바뀌던 진짜 원인은 "피봇이 5분 인덱스라서"가
아니라 **공식 자체가 "인덱스 개수 차이 × 5분"이라는 계산법을 쓰고 있어서**였다 —
피봇 인덱스 표현을 전혀 안 바꾸고, "그 극값이 실제로 언제 갱신됐는지"를 타임스탬프로
같이 들고 있다가 타임스탬프 차이를 쓰면 되는 문제였다.

**결론**: A는 피봇 인덱스 표현을 전혀 건드리지 않고 완전히 독립적으로 적용 가능하다.
B/C를 강제로 끌어들일 필요가 없고, "0단계"라는 개념도 필요 없다. 5m 데이터 완전 제거
(피봇을 1분봉에서 재검출)는 A/B/C 어느 것에도 더 이상 전제조건이 아니므로 맨 마지막으로
미룬다(6장 참고, 착수 안 함).

## 3. A 단계 (적용 완료) — `wave_duration_day`, 타임스탬프 기반

기존 `forming_high`/`forming_low`가 갱신되는 바로 그 순간에 타임스탬프도 같이 기록한다.

```python
# 사전 계산 (ts_1m_ms와 동일한 ns-해상도 가드 방식으로 1회)
ts_5m_ms = (df_5m["ts_dt"].astype("datetime64[ns]").astype("int64") // 10**6).values
```

```python
# idx_5m 경계를 넘을 때 리셋 (기존 forming_high/low 리셋과 함께)
forming_high = 0.0; forming_high_ts_ms = 0
forming_low = 1e18; forming_low_ts_ms = 0
```

```python
# 매 행, 기존 forming_high/low 갱신 조건 그대로에 타임스탬프만 추가
if highs_1m[i] > forming_high:
    forming_high = highs_1m[i]
    forming_high_ts_ms = ts_1m_ms[i]
if lows_1m[i] < forming_low:
    forming_low = lows_1m[i]
    forming_low_ts_ms = ts_1m_ms[i]
```

```python
# wave_duration_day 계산 (extreme_idx는 기존 코드가 이미 계산해주는 값 그대로 재사용)
if extreme_idx == idx_5m:
    a_p1_ts_ms = forming_high_ts_ms if extreme_type == "high" else forming_low_ts_ms
else:
    a_p1_ts_ms = ts_5m_ms[extreme_idx]   # memo_max_h_idx/memo_min_l_idx/p1_idx는 전부
                                          # 이미 5분봉 인덱스라 직접 조회 가능

a_p2_ts_ms = ts_5m_ms[a_p2["index"]]
wave_duration_day = max(0.001, (a_p1_ts_ms - a_p2_ts_ms) / 86_400_000)
```

### 왜 이게 진짜 최소 수정인지

- `a_p1["index"]`/`a_p2["index"]`의 표현 방식(5분 인덱스)을 전혀 안 바꿈 → 그걸 그대로
  쓰는 RSI 다이버전스(`calc_rsi_divergence`), `vol_idx`, `memo_key` 캐시 구조 전부
  **영향 없음**. 피봇 재검출도, 메인 루프 재작성도, RSI 배열 교체도 필요 없음.
- 남는 5m 의존성은 `ts_5m_ms[a_p2["index"]]`/`ts_5m_ms[extreme_idx]`(이미 확정된 5분봉
  타임스탬프 조회)뿐 — 이건 피봇 검출 자체가 여전히 5분봉에서 되고 있는 한 불가피하고,
  그건 6장(맨 마지막, 미착수)의 몫.
- 새로 추가되는 상태는 `forming_high_ts_ms`/`forming_low_ts_ms` 스칼라 2개, 사전 계산
  배열 `ts_5m_ms` 1개뿐. 인과성 위반 없음(둘 다 항상 현재 행 이하의 과거 시각만 가짐).

### 검증 (A 단계)

1. **필드별 회귀**: `wave_duration_day` 딱 한 필드만 바뀌어야 함 — 나머지 전부(RSI
   다이버전스 3종 포함) `np.array_equal`로 완전 동일해야 함.
2. **정상성 확인**: 아직 형성 중인 5분봉 안에서 `wave_duration_day`가 이제 분 단위로
   정확히(레코드가 실제로 갱신된 그 순간 기준) 반영되는지 몇 개 파동에 대해 수동 대조.
3. `distribution_report` 재생성 → `env.py`의 `NORM["duration_log_max"]` p99 커버리지
   재확인(값이 약간 달라지므로).
4. 스모크 캐시(`--recent-days`) 먼저 생성해 위 1~3 검증 후 전체 캐시 생성.

### 버전

`CACHE_VER`: `v9c` → `v9d1`(2026-08-04 확정). 필드명(`DTYPE_V9`)은 동일 유지되므로
`REQUIRED_CACHE_FIELDS` 가드는 이 의미 변경을 못 잡음 — 기존 `v9c` 캐시는 남겨두고
`cache_path_for(cache_ver=...)` 지정 규율 유지.

## 4. B 단계 (적용 완료) — RSI 다이버전스 3종, `5m_rsi`(1분 컬럼) 기반

`*_1m.csv`에 이미 있는 `5m_rsi`(1분 종가 기준 period-70 RSI)를 사용 — Wilder RSI를
처음부터 재계산해 인과성(미래 데이터 미참조) 확인됨.

### 최초안(반려됨): O(1) 러닝 익스트림

A의 `forming_high`/`forming_low` 패턴을 그대로 이식해, 파동 시작(`a_p2` 변경) 시 RSI
극값을 시드값 1개로 초기화하고 매 행 O(1)로 갱신하는 방식이었다. 사용자 검토로 3가지
결함이 드러나 반려:

1. **확정 지연 구간 누락**: `a_p2`는 zigzag 확정 지연(실측 30~214분) 때문에 실제 발생
   시각보다 한참 뒤에야 인지된다. 러닝 방식은 그 지연 구간의 RSI 이력을 스캔 범위에서
   통째로 놓친다.
2. **고점 이후 덮어쓰기**: 기존 로직은 탐색 구간 끝이 `a_p1`(파동의 현재 극점)에 고정돼
   조정 구간에선 탐색이 멈추는데, 러닝 방식은 `a_p1` 정지 여부와 무관하게 매 행 갱신해
   조정 구간의 RSI 스파이크가 진짜 정점 RSI를 덮어쓸 수 있다.
3. **세 필드 간 시점 분열**: `div_price_gap`(`end_price=a_p1["price"]`, 정지되면 자동
   고정)과 러닝 방식의 `div_rsi_gap`/`dist`("지금" 기준으로 계속 이동)가 서로 다른
   시점을 가리키게 된다.

### 최종 설계: 정적 배열 슬라이스 + `wave_duration_day`와 동일한 형성/고정 분기

러닝 상태 대신 매번 `rsi_1m_arr`를 슬라이스해 `argmax`/`argmin`(구 `calc_rsi_divergence`와
동일 정의, 5분봉 대신 1분 배열 스캔). 탐색 시작(`a_p2`)은 파동 시작 행에서 정밀 1분
인덱스를 정적 배열(`ts_5m_high/low_precise_idx1m`)에서 직접 조회 — 확정 지연 구간이
이미 배열에 존재하므로 결함① 없음. 탐색 끝(`a_p1`)은 `wave_duration_day`(A단계)와
정확히 같은 "형성 중이면 라이브, 고정되면 정밀 시각" 분기를 재사용 — `div_price_gap`과
항상 같은 시점을 보게 되어 결함②③ 없음. 탐색 구간 `(a_p2_idx1m, a_p1_idx1m)`이 바뀔
때만 재스캔해 성능 영향 없음(강한 추세 구간에서만 매 행 재계산, 조정 구간은 메모이즈).

`prep_features.py`: 사전계산 확장(`ts_5m_high/low_precise_idx1m`, `rsi_1m_arr`),
`forming_high/low_idx1m` 추가, 구 `div_key`/`div_end_idx`/`calc_rsi_divergence` 호출부
전체 교체, `rsis_5m`(`df_5m["rsi"]`) 의존 제거. `algorithm.py`: `calc_rsi_divergence`
함수 삭제(미사용).

### 검증

풀 캐시(BTC/ETH) 필드별 회귀(바뀐 필드는 `div_rsi_gap`/`dist_from_div_peak_to_end`/
`div_price_gap`/`rsi_now` 4개뿐), 인과성(`dist_from_div_peak_to_end >= 0.001` 항상),
`a_p1` 정지 행(BTC 323만/ETH 330만) 전부에서 세 필드 동시 동결 확인(위반 0), 확정 지연
30~214분짜리 실제 파동에서 탐색 시작점이 지연 구간을 포함함을 별도 스크립트로 확인.
`CACHE_VER`: v9d1 → **v9d2**.

## 5. C 단계 (적용 완료) — `adx_5m`/`atr_5m`/`volatility_ratio`/`relative_volume_strength`

피봇과 무관 — A/B와 순서 상관없이 적용 가능했음. 애초 설계(아래 옛 초안)는 직접
roll/shift로 재계산하는 방식이었으나, **사용자가 `*_1m.csv`에 `5m_adx`/`5m_atr`/
`5m_vol_ratio`/`5m_vol_str` 4개 컬럼을 이미 미리 계산해 추가해둬** 직접 재구현할 필요가
없어짐 — 그 컬럼들을 읽어 쓰는 것으로 스코프가 대폭 축소됨.

인과성은 직접 재구성(`rolling(5)`+`shift(5)` 기반)해 상관계수로 검증: `5m_vol_ratio`
0.999995, `5m_vol_str` 0.9999998, `5m_atr` 0.9999980(사실상 동일), `5m_adx` 0.9795
(ADX가 비선형 재귀라 캔들 구성 디테일 차이가 누적됐을 가능성, 같은 인과적 계열로 판단).

`adx_5m`/`atr_5m`/`volatility_ratio`는 "현재 시장 레짐"을 보는 필드라 원래도 "지금"을
보고 싶었으나 5분봉이라 `prev_5m_idx`(직전 마감 5분봉)로 근사했던 것 — 1분 컬럼은 매분
인과적이므로 `i`(지금) 그대로 조회. `relative_volume_strength`만 성격이 달라(파동 끝
피봇 캔들 자체의 볼륨 백분위, "지금"이 아니라 **a_p1 위치**를 봐야 함) B단계에서 이미
계산해 둔 `a_p1_idx1m`(형성 중이면 라이브, 고정되면 정밀 1분 행)에 그대로 인덱싱 — 1분
배열은 항상 인과적이라 기존 `vol_idx = min(a_p1_idx, prev_5m_idx)` 클램프가 불필요해짐
(제거가 오히려 더 정확 — 형성 중에도 정밀 조회 가능, A/B단계와 같은 개선 패턴).

`prep_features.py`: `df_5m` 기반 계산 블록(`calculate_adx`/`candle_range`+`ambient_vol`/
`rolling_volume_strength` 호출)을 4개 1분 배열 로드로 교체, `vol_idx`/`prev_5m_idx`
삭제, `calculate_adx`/`rolling_volume_strength` 함수와 `VOL_STRENGTH_PERIOD` 상수 삭제
(미사용, 외부 참조 없음 확인).

**검증**: 풀 캐시(BTC/ETH) 필드별 회귀(바뀐 필드 4개만, 나머지 비트 단위 동일), 같은
5분봉 그룹 내에서도 80~98% 행이 값이 갱신됨 확인(기존엔 0%), `relative_volume_strength`
범위 `[0,1]` 확인. `CACHE_VER`: v9d2 → **v9d3**.

### 옛 초안(참고용, 실제로는 위처럼 컬럼을 직접 읽는 것으로 대체됨)

```python
high_roll = high_1m.rolling(5).max()
low_roll  = low_1m.rolling(5).min()
close_roll_prev = close_1m.shift(5)
```
- `adx_5m`/`atr_5m`: 위 roll 시리즈로 TR/+DM/-DM 구성, `calculate_adx`와 동일 RMA를
  `period=70`(14×5)으로 적용.
- `volatility_ratio`: `range_roll = high_roll - low_roll`;
  `ambient_vol = range_roll.shift(5).rolling(50, min_periods=1).mean()`.
- `relative_volume_strength`: `vol_roll = volume_1m.rolling(5).sum()`; 기존
  `rolling_volume_strength`를 `period=2500`으로 적용.

## 6. 5m 데이터 완전 제거 (피봇 1분봉 재검출) — 적용 완료 (2026-08-05)

A/B/C 어느 것에도 더 이상 전제조건이 아니었지만, "이중 인덱스 구조 자체를 없애고 싶다"는
목적으로 최종 착수·완료. 가장 크고 리스크 높은 변경이었던 만큼 계획 단계에서 사용자가
클램프 결함(아래)을 미리 지적해 구현 전에 수정.

### 설계

- `find_dynamic_pivots(df_1m, n=n, min_diff=min_diff)` — `min_diff`는 스케일 불변 그대로,
  `n`은 `PIVOT_PARAMS`에 이미 ×5 반영된 1분봉 기준 값(BTC 3→15, ETH 4→20)을 그대로 사용
  (2026-08-05 리팩터링: 최초엔 호출부에서 매번 `n*5`로 환산했는데, 상수 정의 시점에 미리
  곱해두는 쪽으로 변경 — 결과값 동일, 캐시 재생성 불필요). `find_dynamic_pivots`
  (algorithm.py) 자체는 캔들 단위 무관 범용 함수라 무수정.
- 메인 루프의 `idx_5m`/`memo_key`/`forming_high`/`forming_low`를 `p1_idx` 변경시에만
  리셋되는 `running_extreme_price`/`idx` 증분 추적으로 교체 — 리셋 시 `p1_idx+1..현재`
  구간을 1회 스캔해 시드(확정 지연 구간 포함, B단계에서 검증된 결함 패턴 재발 방지),
  이후 매 행 O(1) 증분. 총 스캔량은 텔레스코핑 합이라 전체 O(total_1m).
- 피봇 인덱스가 곧 1분 행 번호가 되어, A/B단계가 만든 "5분 인덱스 ↔ 1분 정밀 시각/인덱스"
  변환 장치(`ts_5m_*_precise_ms`/`idx1m` 등)가 통째로 불필요해져 삭제 — `ts_1m_ms[a_p1["index"]]`처럼
  직접 조회. `_5m.csv` 로딩 자체 제거(`load_symbol`이 `df_1m`만 반환).

### 🚨 클램프 결함 (구현 전 사용자 지적, 계획 단계에서 수정)

`running_extreme_price`를 클램프 없이 그대로 `extreme_price`에 쓰면, 확정 피봇(p1) 직후
급격한 갭다운/갭업으로 이후 구간의 실제 극값이 `p1["price"]`를 역행할 때(예:
`low=100` 확정 직후 그 뒤 모든 high가 100 미만) "low(100) 다음에 그보다 낮은
high(90)"처럼 **파동 방향이 뒤집히는 구조 파괴**가 생긴다. 구코드는
`max(p1["price"], memo_max_h, forming_high)`/`min(...)`으로 이 클램프를 암묵적으로
하고 있었는데, 새 러닝 익스트림 설계 초안엔 이게 빠져있었음 — 사용 시점에 명시적으로
재현해 수정:
```python
if extreme_type == "high":
    extreme_price = max(p1["price"], running_extreme_price)
    extreme_idx = running_extreme_idx if running_extreme_price > p1["price"] else p1_idx
else:
    extreme_price = min(p1["price"], running_extreme_price)
    extreme_idx = running_extreme_idx if running_extreme_price < p1["price"] else p1_idx
```

### 검증 (A/B/C와 다른 방식 — 피봇 SET 자체가 달라지므로 비트 동일 검증 불가)

1. **피봇 SET 비교**(`np.searchsorted` 벡터화 최근접 매칭 — 최초 나이브 Python 이중루프는
   12분+ 걸려도 안 끝나 벡터화로 교체): BTC 30190↔29438개, ETH 15000↔14786개. 매칭된
   피봇의 가격차 중앙값 0%(대부분 완전 일치), 시간차 중앙값 2분/p95 4분. 아웃라이어
   (가격차>0.01%: BTC 7%/ETH 4%) 중 최악 사례(BTC 14.8% 가격차) 직접 조사 — **2021-09-07
   BTC 플래시 크래시**에서 5분봉 버전은 짧은 반등을 저점(49700)으로 오판 확정했는데,
   1분봉 버전은 그 반등이 n-거리 조건을 못 채워 계속 추적을 이어가 진짜 바닥(42322)까지
   정확히 잡아냄 — 결함이 아니라 1분 정밀도의 정당한 이점으로 확인.
2. **클램프 회귀 검증**: `is_bullish`와 `end_price`/`start_price` 대소관계 일치 어설션,
   풀 데이터셋(BTC 345만/ETH 344만 행) 전부 위반 0건.
3. 기존 인과성 검증(`dist_from_div_peak_to_end>=0.001`, `relative_volume_strength∈[0,1]`,
   같은 파동 내 `wave_duration_day` 단조 비감소) 전부 위반 0건.
4. `distribution_report`가 v9d3과 거의 동일한 스케일 — `NORM` 조정 불필요.
   `env.py` 로드/스텝 스모크 통과. BTC 행수가 v9d3 대비 4행 적음(`WARMUP_5M`→`WARMUP_1M`
   웜업 경계 정의 차이로 데이터 시작부 미세한 갭, 무해).

`CACHE_VER`: v9d3 → **v9d4**(사용자 지정 — 피봇 검출 방식 자체가 바뀌는 질적 전환이지만 v9d 계열 번호로 유지).
`design_spec.md` 반영 완료.
