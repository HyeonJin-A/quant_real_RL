# Quant V9 Standalone Architecture & Specification

## 1. 개요 (Overview)
본 프로젝트(`quant_v9`)는 기존 `quant_with_RL` 프로젝트에 누적된 과거 버전(v1~v8) 코드 및 미사용 대용량 파일들을 제외하고, 메인 로직인 **V9 강화학습 트레이딩 파이프라인**만을 추출하여 독립 구축한 프로젝트입니다.

---

## 2. 프로젝트 디렉토리 구조 (Directory Structure)

```text
c:/workspace/quant_v9/
├── config/                  # 설정 파일 저장소
├── data/
│   └── candle_data/         # BTC-USDT-SWAP, ETH-USDT-SWAP 캔들 데이터 (1m, 5m CSV)
├── caches/                  # V9 전용 피처 캐시 (.npy) 및 분포 리포트 (.json)
├── models/                  # 학습된 PPO 모델 파일 (.zip) 및 메타 정보 (.json)
├── logs/                    # TensorBoard 학습 로그
├── src/                     # 핵심 V9 알고리즘 및 강화학습 파이프라인
│   ├── __init__.py
│   ├── algorithm.py         # N-Bar 기반 동적 파동(Pivot) 추출 알고리즘
│   ├── prep_features.py     # 1m/5m 캔들 기준 V9 피처 추출 및 캐시 생성기
│   ├── env.py               # Gymnasium 기반 V9 Semi-MDP Adaptive 트레이딩 환경
│   ├── train.py             # SB3 PPO 학습 및 LogStdClampCallback 등 폭주 방지 제어
│   └── eval.py              # 백테스트 및 70/15/15 분할 구간 검증 하네스
├── V9 Design.md             # V9 핵심 알고리즘/피처/보상 상세 설계 문서
├── requirements.txt         # 의존성 라이브러리 목록
└── design_spec.md           # 통합 설계 및 마이그레이션 사양서 (본 문서)
```

---

## 3. 핵심 모듈 사양 (Core Component Specifications)

### 3.1 [algorithm.py](file:///c:/workspace/quant_v9/src/algorithm.py)
* **`find_dynamic_pivots(df, n, min_diff)`**: 캔들 데이터에서 가격 변동률(`min_diff`)과 간격(`n`) 조건에 따른 고점/저점 피봇을 실시간 검출하여 파동을 형성.

### 3.2 [prep_features.py](file:///c:/workspace/quant_v9/src/prep_features.py)
* **피처 추출 엔진**: 파동 정보, ADX/ATR (RMA 방식), RSI Divergence, relative_volume_strength, fib_pos 등의 14차원 피처를 계산하여 `caches/features_v9_{SYMBOL}.npy` 저장.
* **사용법**: `python src/prep_features.py --symbols BTC-USDT-SWAP ETH-USDT-SWAP`

### 3.3 [env.py](file:///c:/workspace/quant_v9/src/env.py)
* **`TradingEnvV9`**: Gymnasium 트레이딩 환경.
* **`exit_mode="adaptive"`**: semi-MDP 구조로, 진입 선택 시 레버리지(1~50배), 손절폭, 반익/완익절을 동적 선택.
* **`dynamic_leverage_ceiling`**: ATR% 기반 종목 독립적 동적 레버리지 제한 적용.

### 3.4 [train.py](file:///c:/workspace/quant_v9/src/train.py)
* **PPO 학습 파이프라인**: `LogStdClampCallback`, `LeverageMaxSchedule`, `explore_bonus` 커리큘럼 적용으로 연속 액션 표준편차 폭주 방지.
* **사용법**: `python src/train.py --seed 0`

### 3.5 [eval.py](file:///c:/workspace/quant_v9/src/eval.py)
* **백테스트 및 지표 판정**: train 70% / valid 15% / test 15% 시계열 구간 검증, PnL, WinRate, MSL, V8 Score 산출.
* **사용법**: `python src/eval.py --model models/v9_ppo_seed0_best.zip --split test`
