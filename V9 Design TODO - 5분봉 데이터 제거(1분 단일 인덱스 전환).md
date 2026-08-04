# V9 Design TODO — 5분봉 데이터 제거, 1분 단일 인덱스 아키텍처 전환

상태: **A 단계 적용 완료 및 검증 통과 (2026-08-04, `CACHE_VER=v9d1`).** B/C는 미착수,
5m 데이터 완전 제거(피봇 구조 개편)는 맨 마지막으로 이동.

**적용 결과 요약**: `src/prep_features.py` 3곳 수정(설계 결함 재점검 후 확정된 버전, 3장
참고). 풀 캐시(BTC/ETH) 재생성 후 필드별 회귀 — `wave_duration_day` 딱 한 필드만 값이
다르고 나머지 전부 비트 단위 동일. 같은 파동 유지 구간 내 단조 비감소 검증: ETH는 342만
행 전부 통과, BTC는 345만 행 중 1행(2020-05-13 16:35 5분봉)에서 위반 발견 → 추적 결과
**이번 변경이 만든 결함이 아니라 원본 데이터 자체의 결함**(그 5분봉의 `low` 컬럼이 실제
1분봉 5개의 최저가와 불일치, 5m/1m CSV 간 원본 데이터 불일치, 발생빈도 1/345만) — 기존
(미변경) `memo_min_l`이 그 오차를 그대로 드러낸 것뿐이라 이번 수정 범위 밖으로 판단,
기록만 남기고 조치 안 함. `distribution_report` p99(BTC 1.05일/ETH 1.81일)도
`NORM["duration_log_max"]` 커버리지 이내라 `env.py` 변경 불필요. `design_spec.md` 반영 완료.

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

## 3. A 단계 (지금 적용) — `wave_duration_day`, 타임스탬프 기반

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

## 4. B 단계 — RSI 다이버전스 3종 (미착수, A와 무관하게 재검토 필요)

A의 교훈(인덱스 표현을 안 바꾸고도 해결 가능했던 것)이 B에도 적용되는지 별도로 다시
검토해야 함 — 예단하지 않음. `rsi_now`/`div_rsi_gap`/`dist_from_div_peak_to_end`가 실제로
5m_rsi 배열이 꼭 필요한지, 아니면 A처럼 인덱스 표현은 그대로 두고 다른 방식으로 풀 수
있는지부터 다시 확인하고 착수.

## 5. C 단계 — `adx_5m`/`atr_5m`/`volatility_ratio`/`relative_volume_strength` (미착수, 독립적)

피봇과 무관 — A/B와 순서 상관없이 언제 적용해도 됨. 설계는 기존 그대로 유효:

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
- 필드명 그대로 유지, 주석만 갱신.

## 6. 5m 데이터 완전 제거 (피봇 1분봉 재검출) — 맨 마지막, 착수 안 함

A/B/C 어느 것에도 더 이상 전제조건이 아님이 확인됨 — 순수하게 "이중 인덱스 구조 자체를
없애고 싶다"는 코드 정리 목적일 때만 의미가 있는, 가장 크고 리스크 높은 변경. 필요성이
분명해지기 전까지는 계획만 남겨두고 착수하지 않는다.

- `find_dynamic_pivots(df_1m, n=n_5m*5, min_diff=min_diff_5m)` — `min_diff`는 스케일
  불변 그대로, `n`만 ×5(BTC 3→15, ETH 4→20).
- 메인 루프의 `idx_5m`/`memo_key`/`forming_high`/`forming_low`를 `p1_idx` 변경시에만
  리셋되는 `running_extreme_price`/`idx` 증분 추적으로 교체.
- 이 변경은 피봇 SET 자체가 미세하게 달라질 수 있어(1분 단위 wick 가시성) 전 필드 재검증
  필요 — 기존 5분봉 지그재그 대비 피봇 개수·가격·시각 차이 리포트로 확인 후 진행.
