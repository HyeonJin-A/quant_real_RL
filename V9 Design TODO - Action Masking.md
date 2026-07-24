# V9 Design TODO — rl 모드 Action Masking

상태: **설계 단계, 미구현.** 코드 변경 없음. 2026-07-23 대화에서 나온 아이디어를 문서화.

## 1. 배경

rl 모드(`exit_mode="rl"`, `Discrete(3)` {Hold, Enter, Close})는 상태에 따라 일부 행동이 무효(no-op)다:
- 무포지션일 때 Close → Hold와 완전히 동일하게 처리
- 보유중일 때 Enter → Hold와 완전히 동일하게 처리 (`env.py` `_step_rl`, "상태와 안 맞는 조합... 전부 그대로 통과")

지금은 이 무효 조합을 **조용히 Hold로 흡수**하는 방식이라 다음 두 가지 비용이 있다:
1. **엔트로피 지표 해석 왜곡**: 이번 세션 내내 `entropy_loss`를 붕괴 진단(과매매/무행동)의 핵심 근거로 썼는데, 무효 조합끼리는 확률을 나눠 가져도 행동 다양성엔 전혀 기여하지 않아 지표가 실제 행동 다양성과 어긋날 수 있음.
2. **행동공간 일부가 항상 낭비**: 어느 상태에서든 3개 중 1개는 항상 다른 것과 완전히 같은 결과라 정보량이 100% 효율적이지 않음.

이걸 구조적으로 제거하는 방법이 **action masking** — 상태에 안 맞는 행동을 후보에서 물리적으로 제거(마스킹)하는 것.

## 2. 필요 요소

- **의존성 추가**: `sb3-contrib` (`MaskablePPO`, `ActionMasker`) — 현재 미설치, `requirements.txt`에 없음. stable-baselines3 자체 PPO는 마스킹 미지원.
- **`env.py`**: `TradingEnvV9`에 `action_masks()` 메서드 추가 (rl 모드 전용) — 현재 상태 기준 유효 행동 boolean 배열 반환.
  ```python
  def action_masks(self):
      # 예시 스케치 — 정확한 시그니처는 sb3-contrib 버전 문서 확인 필요
      if self.exit_mode != "rl":
          return None  # 또는 전부 True (마스킹 불필요 모드)
      if self.pos_dir == 0:
          return [True, True, False]   # Hold, Enter 가능 / Close 불가
      else:
          return [True, False, True]   # Hold, Close 가능 / Enter 불가
  ```
- **`train.py`**:
  - `SubprocVecEnv`/`DummyVecEnv` 생성 시 각 env를 `sb3_contrib.common.wrappers.ActionMasker`로 감싸야 함 (또는 SB3 계열 마스킹 래퍼 규약에 맞춰 `action_masks` 콜러블 등록).
  - `PPO(...)` → `MaskablePPO(...)`로 교체 (rl 모드일 때만 분기 필요 — adaptive/rule은 기존 PPO 유지).
  - 콜백들(`EntCoefSchedule`, `ExploreBonusSchedule` 등)이 `MaskablePPO`와 호환되는지 확인 필요 (SB3 콜백 API 대부분 호환되지만 실제 검증 필요).
- **`eval.py`**: `model.predict(obs, deterministic=True)` 호출부를 `MaskablePPO` 로드 시 `action_masks` 인자를 넘기도록 수정 (`run_policy_on_range` 등 rl 모드 추론 경로 전체 점검).

## 3. 적용 범위

- **rl 모드 전용.** adaptive(Box 연속 액션)·rule(Discrete(2), 이미 상태 무관 완전 유효)은 이 문제 자체가 없어 변경 불필요.
- 기존 rl 모드 체크포인트(`v9_fullctrl_*`)는 `MaskablePPO`와 정책 클래스가 달라 **호환 안 됨** — 마스킹 도입 시 처음부터 재학습 필요.

## 4. 예상 효과 (검증 전 가설)

- entropy_loss가 실제 행동 다양성만 반영하게 되어, 이번 세션에서 반복했던 "explore_bonus/ent_coef 타이밍" 진단이 더 정확해질 가능성.
- 무효 행동에 낭비되던 학습 신호가 유효 행동 쪽으로만 집중되어, 수렴이 다소 빨라지거나 안정될 가능성.
- 다만 이건 가설이며, 지금까지 겪은 과매매/무행동 붕괴의 **근본 원인(explore_bonus 크기, ent_coef 타이밍)과는 별개 축**이라 마스킹만으로 그 문제들이 해결된다는 보장은 없음 — 오히려 지금 확정된 explore_bonus=0.001 베이스라인 위에 안전하게 얹을 수 있는지부터 확인 필요.

## 5. 리스크 / 열린 질문

- `sb3-contrib` 버전이 현재 `stable-baselines3==2.7.1`/`torch==2.8.0` 핀과 호환되는지 확인 필요 (버전 매트릭스 점검 선행).
- 마스킹 도입이 기존 콜백(특히 `ValidationCallback`의 `run_policy_on_range` 추론 경로)에 미치는 영향 전수 점검 필요 — 스모크 테스트(2만 스텝) 선행 후 본 학습.
- 우선순위: 지금 진행 중인 leverage=3 실험, explore_bonus 등 이미 검증된 축이 안정화된 뒤 착수하는 게 안전 — 여러 변수를 동시에 바꾸면 원인 격리가 어려워짐(이번 세션에서 반복 확인된 교훈).

## 6. 구현 순서 제안

1. `sb3-contrib` 설치 가능/버전 호환 여부 확인.
2. `env.py`에 `action_masks()` 추가, 단위 테스트로 상태별 마스크 값 검증.
3. `train.py`에 `MaskablePPO` 분기 추가 (rl 모드일 때만), 스모크 테스트(2만 스텝, `--dummy-vec`).
4. `eval.py` 추론 경로에 마스킹 반영, 기존 rl 베이스라인과 나란히 비교할 수 있게 별도 run_name으로 학습.
5. 결과가 기존 베이스라인(explore_bonus=0.001, leverage=1, sel_monthly_log -0.0174)을 능가하는지로 채택 여부 판단.
