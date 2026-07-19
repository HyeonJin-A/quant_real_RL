# <span style="color: #FFFFFF;">Quant V9 — 시스템 설계서 (design_spec)</span>

본 프로젝트(`quant_real_RL`)는 5m 파동(피봇) 기반 **역추세(fade) 진입을 PPO 정책이 학습하는 V9 강화학습 트레이딩 파이프라인**의 독립 저장소입니다. 기존 `quant_with_RL`에서 V9 메인 로직만 추출해 구축했습니다.

현재 아키텍처는 **`adaptive` semi-MDP 구조**입니다: 정책은 무포지션 결정 시점에만 행동하며, 진입 순간에 레버리지/손절폭/반익절/완익절을 전부 확정하고, 청산까지는 numba 시뮬레이터가 fast-forward하여 실현 손익을 그 스텝의 보상으로 즉시 반환합니다 (결정 1개 = 거래 1개 = 완결된 보상). 초기 설계(`V9 Design.md`)의 풀 컨트롤("rl" 모드)은 "아무것도 안 함" 붕괴로 폐기되었고, 본 문서가 현행 사양의 기준입니다.

*최종 갱신: 2026-07-19*

---

## <span style="color: #FFA500;">1. 디렉토리 구조 및 데이터</span>

### <span style="color: #2E8B57;">디렉토리 구조</span>
```text
quant_real_RL/
├── data/candle_data/        # OKX SWAP 캔들 CSV (1m/5m)
├── caches/                  # V9 피처 캐시(.npy) + 분포 리포트(.json)
├── models/                  # PPO 체크포인트(.zip) + best 메타(.json)
├── logs/                    # TensorBoard 학습 로그
├── src/
│   ├── algorithm.py         # N-Bar 동적 피봇(파동) 검출 + 보조 지표 함수
│   ├── prep_features.py     # 1m 행 단위 인과적 피처 캐시 생성기
│   ├── env.py               # TradingEnvV9 (Gymnasium, semi-MDP adaptive)
│   ├── train.py             # SB3 PPO 학습 + 커리큘럼/안정화 콜백
│   └── eval.py              # 백테스트 하네스 (70/15/15 분할, 합격 기준 판정)
├── V9 Design.md             # 초기 설계서 (풀 컨트롤 시절 — 역사 참고용)
├── V9 Issue - *.md          # 미해결 이슈 노트 (7장 참조)
└── design_spec.md           # 현행 사양서 (본 문서)
```

### <span style="color: #2E8B57;">사용 데이터</span>
| 심볼 | 기간 (1m) | 행 수 (1m) | 용도 |
|---|---|---|---|
| BTC-USDT-SWAP | 2019-12-16 ~ 2026-07-11 | ~345만 | 학습 + 검증 + 테스트 (모델 선택 기준 심볼) |
| ETH-USDT-SWAP | 2019-12-25 ~ 2026-07-12 | ~344만 | 학습 + 검증 + 테스트 (지표 기록은 유지, 모델 선택에선 제외) |
| SOL-USDT-SWAP | 2021-01-22 ~ 2026-07-12 | ~288만 | 미학습 (완전 미학습 종목 일반화 테스트용 예비) |

- 5m CSV는 동일 기간의 파동 감지/지표 계산용. CSV는 시간 내림차순으로 저장되어 있으며 로드 시 오름차순 정렬 + OHLC 결측 행 제거.
- **관측에 심볼 ID는 넣지 않습니다** — 종목 특수 패턴 암기를 차단하고 두 종목에 동시에 통하는 패턴만 학습하도록 강제. 코드 어디에도 종목 분기가 없습니다.

### <span style="color: #2E8B57;">주요 의존성 (requirements.txt 핀 고정)</span>
`stable-baselines3==2.7.1`, `gymnasium==1.1.1`, `torch==2.8.0`, `numba==0.60.0`, `numpy==2.0.2`, `pandas==2.3.3`

---

## <span style="color: #FFA500;">2. 피처 파이프라인 (`algorithm.py` → `prep_features.py`)</span>

### <span style="color: #2E8B57;">파동 감지 (`find_dynamic_pivots`)</span>
- N-Bar 동적 피봇: 가격이 직전 피봇에서 `min_diff` 이상 반전하고, 해당 캔들이 전후 `n`개 캔들의 극점일 때 피봇 확정. 확정 피봇 사이 구간이 "파동"이 됩니다.
- **심볼별 피봇 파라미터** (2026-07-17 그리드서치로 확정, `PIVOT_PARAMS`):
  | 심볼 | n | min_diff |
  |---|---|---|
  | BTC-USDT-SWAP | 3 | 0.0075 |
  | ETH-USDT-SWAP | 4 | 0.015 |
  | (그 외 폴백) | 8 | 0.02 |

### <span style="color: #2E8B57;">피처 캐시 (`features_v9_{SYMBOL}.npy`, 행당 21컬럼)</span>
- 5m 파동을 1m 스냅샷에 매핑. **매 1m 행은 그 시점에 실시간으로 알 수 있었던 정보만 포함** (인과성 보장): forming 캔들 극값 추적, 피봇 확정 지연(`index ≤ idx_5m − n`), 지표는 직전 마감 5m 캔들 기준.
- 주요 컬럼: 파동 방향/시작·끝 가격/스케일/지속시간, RSI 다이버전스(발생 플래그 + rsi_previous + 가격 갭), relative_volume_strength(트레일링 500캔들 min-max 백분위 — V8의 0.0 하드코딩 버그 수정판), volatility_ratio, adx_5m/atr_5m(RMA 방식), fib_pos(파동 내 되돌림 위치), pre_hit_0382(0.382 선터치 플래그), wave_age_min.
- 캐시 생성 시 심볼별 **피처 분포 리포트**(퍼센타일 표 JSON)를 함께 저장 — 정규화 클립 상수가 BTC/ETH 분포를 커버하는지 검증용.
- 사용법: `python src/prep_features.py --symbols BTC-USDT-SWAP ETH-USDT-SWAP` (스모크: `--recent-days 120`)

### <span style="color: #2E8B57;">정규화 규칙 (누출 금지 원칙)</span>
- 모든 관측 피처는 **사전 확정 고정 상수**(`env.NORM`)로만 정규화. 데이터셋 통계(퍼센타일/평균/표준편차) 사용 금지 — train/test 누출 방지 + 라이브 재현성.
- 꼬리 긴 피처(duration, vol_ratio)는 `log1p` 후 클립, 바운디드 지표(RSI, ADX)는 `/100`, ATR은 `atr/close×100`(%) 형태로 스케일 불변화.

---

## <span style="color: #FFA500;">3. 학습 환경 (`env.py` — TradingEnvV9)</span>

### <span style="color: #2E8B57;">exit_mode 3종과 현행 모드</span>
| 모드 | 행동 공간 | 상태 | 비고 |
|---|---|---|---|
| **adaptive** (현행 기본) | Box(-1,1)^6 연속 | **주력** | 진입 시 레버리지/손절/반익/완익까지 정책이 전부 결정, semi-MDP |
| rule | Discrete(2) Skip/Enter | 레거시 | 청산은 V8 엔진(반익절/본절이동/ATR손절) 고정 — Fallback B |
| rl | Discrete(4) 풀 컨트롤 | 폐기 | 2026-07-16 "거래 0건" 붕괴 확인, 코드만 유지 |

### <span style="color: #2E8B57;">관측 공간 (14차원, `_build_static_obs`에서 전 행 사전 정규화)</span>
| # | 피처 | 정규화 |
|---|---|---|
| 1 | wave_scale_percent | clip 0~12% → /12 |
| 2 | wave_duration_day | log1p, clip 0~10일 |
| 3 | has_rsi_divergence | 0/1 |
| 4 | 방향조정 RSI (`rsi if bullish else 100−rsi`) | /100 |
| 5 | divergence_price_gap_percent | clip 0~5% → /5 |
| 6 | volatility_ratio | log1p, clip 0~6 |
| 7 | relative_volume_strength | clip 0~1 |
| 8 | adx_5m | /100 |
| 9 | atr/close % | clip 0~3% → /3 |
| 10 | is_bullish | ±1 |
| 11 | fib_pos (되돌림 위치) | clip −1~2 → /2 |
| 12 | pre_hit_0382 | 0/1 |
| 13–14 | fib_pos 델타 (5분 전/15분 전 대비, 파동 교체 시 0) | clip ±1 |

- 포지션 상태 4칸은 2026-07-19 제거 — semi-MDP는 결정 시점에 항상 무포지션이라 영원히 0인 죽은 차원이었음 ("rl" 모드 전용 18차원으로만 유지).
- wave_age_min은 wave_duration_day와 상관 0.87로 중복이라 관측에서 제외 (델타 피처의 내부 판정에만 사용).

### <span style="color: #2E8B57;">행동 공간 (adaptive, Box(-1,1)^6)</span>
| # | 의미 | 매핑 범위 |
|---|---|---|
| a[0] | 진입 여부 (>0이면 Enter) | — |
| a[1] | 레버리지 | 정수 1 ~ min(커리큘럼 상한, 동적 상한) — 절대 상한 50 |
| a[2] | 손절폭 (ATR 배수) | 0.0 ~ 3.0 (하한 0.0 = 파동 극점 재터치 시 칼손절) |
| a[3] | 반익절 레벨 | 0.1 ~ 0.5 (파동 되돌림 비율) |
| a[4] | 완익절 추가폭 | 완익 레벨 = 반익 레벨 + (0.0 ~ 2.0) — 항상 반익 이상 |
| a[5] | 반익 사용 여부 (>0이면 실행) | 반익 선택제 (2026-07-19) |

- 진입 방향은 학습하지 않음: **항상 파동 역추세(fade)** 고정 (V8과 동일).
- 청산 순서(`simulate_adaptive_exit`, numba): 반익절(50%, a[5] 선택 시) → 완익절(전량) → 고정 손절(진입 후 불변, 본절이동 없음) → 타임아웃(7일, `DEFAULT_MAX_HOLD_BARS=10080`) 종가 정산.

### <span style="color: #2E8B57;">시장 메커니즘 (env에는 "시장의 물리 법칙"만)</span>
- 증거금 **100 USDT 고정** × 정책 선택 레버리지. 사이즈가 아닌 레버리지를 다이얼로 쓰는 이유: 둘 다 달러 손익 스케일엔 동일하게 작용하지만 **청산까지의 거리는 레버리지만이 결정**하기 때문.
- 수수료 0.05%/leg (진입/청산 각각). 강제청산가는 레버리지·수수료만으로 결정되는 가격 비율 — 손절선은 이 가격을 넘지 못하도록 클램프.
- 보상 = **실현 순손익 / 100** 그대로 (+학습 커리큘럼의 explore_bonus만). 보상 클리핑·근접청산 벌점 등 셰이핑은 전부 제거됨 (5장 이력 참조).
- 에피소드: train 구간 내 랜덤 시작 **60일**(86,400행) 청크 (2026-07-19: 30일→60일 확대). 평가는 `fixed_full_range=True`로 분할 구간 전체 단일 연속 롤아웃.

### <span style="color: #2E8B57;">동적 레버리지 상한 (`dynamic_leverage_ceiling`) 🚨</span>
```
leverage_max = clip( 20 / (gap% + ATR% × 선택한 sl_multiplier + 0.1), 1, 50 )
  gap% = |진입가 − 파동극점| / 진입가 × 100   (손절 기준점이 극점이므로)
  0.1  = 수수료 마진(%p),  분자 20 = MAX_TRADE_LOSS_PCT (거래당 손실 예산 %)
```
- **한 거래 최대 손실 ≤ 증거금의 20%를 수학적으로 보장** — 근접청산(-95)과 강제청산이 구조적으로 불가능. 검증: 최대 레버리지 몰빵 랜덤 정책 30,000거래에서 최악 손실 정확히 -20.0, 근접청산 0건.
- 정책이 실제로 고른 sl_multiplier로 상한 계산 → 타이트한 손절일수록 높은 레버리지가 열리는 "손절 거리 기반 사이징". 좋은 타점(gap≈0)+타이트 손절이면 50배까지 개방.
- BTC/ETH **완전 동일 공식, 종목 분기 없음** — ETH의 구조적으로 높은 ATR%가 자연히 더 보수적인 상한으로 이어짐 (BTC/ETH 괴리 이슈 대응). 평가/실전에서도 상시 적용.

---

## <span style="color: #FFA500;">4. 신경망 및 학습 프로토콜 (`train.py`)</span>

### <span style="color: #2E8B57;">신경망 및 PPO 하이퍼파라미터 (stable-baselines3)</span>
| 항목 | 값 | 비고 |
|---|---|---|
| 정책망 | MlpPolicy `net_arch=[64, 64]` | 14차원 관측 + 6차원 연속 액션 (diagonal Gaussian) |
| learning_rate | 3e-4 → 0 선형 감쇠 | |
| n_steps / batch_size / n_epochs | 2048/워커, 512, 10 | |
| gamma / gae_lambda | 0.999 / 0.95 | 1m 스텝 기준 유효 horizon ~16시간 |
| clip_range / vf_coef | 0.2 / 0.5 | |
| ent_coef | 0.01 → 0.005 | 70% 지점까지 0.01 고정 유지 후 선형 감쇠 |
| device | **cpu 강제** | 환경 간 연산 재현성 (GPU 미활용) |
| 총 스텝 / 워커 | 기본 50M / SubprocVecEnv 4 (심볼별 균등 배분) | |

### <span style="color: #2E8B57;">학습 안정화 콜백 (붕괴 이력 3건에 대한 대응 장치)</span>
| 콜백 | 역할 |
|---|---|
| **LogStdClampCallback** | 매 롤아웃 시작 시 policy.log_std를 [-3.0, 1.0](std≈0.05~2.7)로 강제 clamp — 연속 액션 엔트로피 무제한 상방으로 인한 **표준편차 폭주 방지의 핵심**. 평가는 deterministic이라 판단력엔 영향 없음 |
| EntCoefSchedule | ent_coef 0.01을 70% 지점까지 고정 유지 후 0.005로 감쇠 |
| ExploreBonusSchedule | Enter 시 임시 보너스 0.15 → 50% 지점에서 0 (콜드스타트 함정 돌파용, cum_pnl/평가엔 미반영) |
| LeverageMaxSchedule | 레버리지 상한 10 → 30% 지점까지 50으로 선형 확대 (초반 무작위 탐험의 손실 분산 억제) |
| ValidationCallback | 50만 스텝마다 검증셋 전체 결정론적 롤아웃 → TensorBoard 기록 + best 저장 (5장 모델 선택 기준) |
| CheckpointCallback | 100만 스텝마다 체크포인트 |

- ⚠️ 커리큘럼 갱신은 반드시 `VecEnv.env_method("set_curriculum", ...)` 사용 — `set_attr`은 Monitor 래퍼 표면에 그림자 속성만 만들고 내부 env에 안 닿는 SB3 버그 있음 (2026-07-17 실측 확인).
- 사용법: `python src/train.py --seed 0` (스모크: `--timesteps 30000 --workers 2 --dummy-vec --cache-suffix _recent120d`)

---

## <span style="color: #FFA500;">5. 평가 및 모델 선택 (`eval.py`)</span>

### <span style="color: #2E8B57;">데이터 분할 및 평가 방식</span>
- 시계열 70/15/15 (train/valid/test), 날짜 경계는 두 심볼 공통. **테스트셋은 모든 개발 종료 후 단 1회만 평가.**
- 평가는 분할 구간 전체 **단일 연속 롤아웃** (2026-07-19: 16분할 병렬은 경계 오차 ~8-12% 실측으로 폐기).
- 지표: 거래수, 월평균 거래수, 승률, 총 PnL, 평균 PnL, MSL, PnL 표준편차, PF, top1_share, MDD, near_liquidation 건수, V9 Score, 연도별 분해, 수수료 민감도(`--fee`).

### <span style="color: #2E8B57;">V9 Score와 모델 선택 기준 (2026-07-18~19 확정)</span>
```
v9_score = (total_pnl − top1_수익) + win_rate × 1000
모델 선택 점수 = BTC 검증구간 4분할 v9_score의 (평균 − 표준편차)
```
- top1 제외 총수익: "대박 한 방" 의존 체크포인트의 고득점 차단 (합격 기준 ③의 점수화).
- BTC 4분할 평균−표준편차: 특정 레짐에서만 버는 체크포인트가 1년 합계로 위장하는 것을 배제 (검증 +1,412 → 테스트 -242 실측이 계기). 고르게 버는 체크포인트 우대.
- 모델 선택은 **BTC 단독** (2026-07-19): min(BTC,ETH) 기준은 만성적으로 ETH에 끌려 내려가 BTC에서 잘하는 체크포인트를 놓치는 문제. ETH는 TB 지표 기록만 유지.
- near_liq(pnl≤-95) 지표는 **울타리 검증용 경보기** — 동적 레버리지 상한이 정상이면 항상 0이어야 하며, 0이 아니면 상한 공식 버그.
- V8 Score는 완전 폐기 (2026-07-19).

### <span style="color: #2E8B57;">복리 평가 지표 (`compound_metrics`, 2026-07-19 추가 — 평가 전용)</span>
- 실전 운용 방식 기준: **시작 자본 100 USDT, 매 거래 전재산을 증거금으로 투입**하는 복리 시뮬레이션. 산출: 최종 자본, 성장 배수(multiple), 월평균 성장 배수, 복리 자본곡선 MDD(%), 파산 여부.
- **학습 env/보상은 기존 그대로 고정 100 USDT 진입** — 복리는 평가 리포트와 검증 TB 기록(`valid/{SYM}/compound_multiple`, `compound_mdd_pct`)에만 존재하며, 모델 선택 기준에도 미사용.
- 별도 재시뮬레이션 없이 고정 사이징 거래 목록에서 **정확히** 재구성: 손익·수수료가 증거금에 선형 비례(`pos_size = 증거금×lev/price`)라 거래 수익률 `r = pnl/100`이 증거금 규모와 무관하고, semi-MDP라 심볼 내 거래가 겹치지 않아 순차 재투자(`equity ×= 1 + r`) 가정이 정확히 성립. 심볼 간에는 거래가 겹칠 수 있어 **복리 지표는 심볼별로만 산출** (합산 리포트엔 없음).

### <span style="color: #2E8B57;">사전 확정 합격 기준 (심볼별 각각 적용, 변경 금지) 🚨</span>
| # | 기준 |
|---|---|
| ① | 테스트 구간 월평균 3회 이상 거래 |
| ② | 최대 단일 수익 ≤ 총수익의 30% |
| ③ | Top-1 수익 거래 제거 후에도 총 PnL > 0 |
| ④ | 테스트 v9_score ≥ 베이스라인 (`--baseline-score` 입력 시) |
| ⑤ | 3시드 중 2시드 이상 ①~③ 통과 (시드별 리포트 취합 후 별도 판정) |

- 사용법: `python src/eval.py --model models/v9_ppo_seed0_best.zip --split valid` (exit_mode는 학습 때와 일치 필수, 기본 adaptive)

---

## <span style="color: #FFA500;">6. 주요 의사결정 이력 (날짜순)</span>

| 날짜 | 결정 | 근거 |
|---|---|---|
| 07-16 | "rl" 풀 컨트롤 폐기 → semi-MDP(rule → adaptive) 전환 | 검증 4회 연속 거래 0건 붕괴. ent_coef 상향/에피소드 단축 모두 무효 — Hold의 구조적 우위(진입=확실한 수수료 비용) |
| 07-16 | LogStdClampCallback 도입, ent_coef 0.03→0.01 | entropy_loss -7→-62는 "확신 굳음"이 아니라 **std 폭주**였음 (SB3 소스로 부호 재확인, 오진단 정정) |
| 07-16 | explore_bonus 0.05→0.15, LeverageMaxSchedule 추가 | 보너스가 고레버리지발 손실 1건에 묻힘 → 인센티브+탐험 공간 제한 병행 |
| 07-17 | 레버리지 절대 상한 100→50 | 근접청산 62건 전부 45배 이상, 89%가 정확히 100배 — 실측 기반 하향 |
| 07-17 | set_attr → env_method 교체 | SB3 그림자 속성 버그로 커리큘럼이 실제로는 안 먹히고 있었음 |
| 07-17 | 심볼별 피봇 파라미터(n/min_diff) 확정 (BTC 3/0.75%, ETH 4/1.5%) | 그리드서치 |
| 07-18 | 동적 레버리지 상한 도입 (ATR% 기반) | BTC/ETH 괴리의 원인: ETH의 높은 ATR%로 손절이 청산가에 눌림. 종목 고정값 대신 종목 분기 없는 동적 공식 |
| 07-19 | 상한 공식 2차 수정: gap 항 추가 + 분자를 손실 예산(20)으로 | ATR 항만으론 -100 경로 잔존, 갭 항 추가 후에도 정상 손절 -95 실측 → 예산 기반으로 교체 |
| 07-19 | 보상 클리핑(+30) 제거 | 수익 원천인 "소수의 큰 익절" 유인을 클립이 제거 → TP를 짧게 당기는 음의 기대값으로 적응하는 부작용 실측 |
| 07-19 | 근접청산 벌점(-100) 제거, 포지션 상태 관측 4칸 제거 | 둘 다 구조 변경으로 죽은 코드가 됨 (사후 벌점 → 사전 차단 일원화) |
| 07-19 | 반익 선택제(a[5]), SL 하한 0.3→0.0, 모델 선택 BTC 단독 + 4분할 | 진입 시점 결정의 자유도 확대 + 레짐 위장 차단 |
| 07-19 | 에피소드 길이 30일→60일 | 학습 에피소드당 샘플 다양성/거래 수 확대 |
| 07-19 | 복리 평가 지표 추가 (`compound_metrics`) | 실전은 복리 운용이므로 전액 재투입 기준 성장률/MDD를 평가·검증 로그에 병기 (학습·모델 선택은 고정 100 USDT 유지) |

---

## <span style="color: #FFA500;">7. 미해결 이슈 (이슈 노트)</span>

- **완익 사전결정 구조의 한계** (`V9 Issue - 완익 사전결정 구조.md`, 미구현): 진입 순간에 완익 목표를 박제 → 보유 기간(수 시간~수 일)의 신규 정보를 전부 버림. 테스트 top1_share 55.2%와 무관하지 않을 가능성. 최우선 후보안은 "주기적 재결정 포인트"(N시간마다 유지/청산/목표수정 — semi-MDP 골격 유지). 어떤 안이든 행동 공간이 바뀌어 기존 체크포인트와 비호환.
- **BTC/ETH 성과 괴리** (`V9 Issue - BTC,ETH괴리잡기.md`, 부분 대응): 동적 레버리지 상한으로 근접청산 격차는 구조적으로 해소했으나, ETH 자체의 수익성 문제는 미해결 — 모델 선택을 BTC 단독으로 돌린 것이 그 방증. ETH는 현재 지표 관찰만 유지.

---

## <span style="color: #FFA500;">8. 산출물 명명 규칙</span>

- 피처 캐시: `caches/features_v9_{SYMBOL}[_recent{N}d].npy` + `dist_report_v9_{SYMBOL}[...].json`
- 모델: `models/v9_ppo_seed{S}_{steps}_steps.zip` (주기 체크포인트), `..._best.zip` + `..._best_info.json` (검증 best), `..._final.zip` (학습 종료 시점)
- 로그: `logs/v9_ppo_seed{S}_*/` (TensorBoard — `valid/{SYMBOL}/...` 지표군, `valid/BTC-USDT-SWAP/v9_score_sel`이 모델 선택 점수)
