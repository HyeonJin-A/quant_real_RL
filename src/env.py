"""
V9 트레이딩 환경 (V9 Design.md 5~6장, 9장)

두 가지 exit_mode를 지원한다:

- exit_mode="rl" (보유 중 풀 컨트롤): 결정 주기 매 1m 행, 행동 Discrete(3)
  {0: Hold, 1: Enter, 2: Close}. 보상은 매 스텝 mark-to-market 손익 변화량(dense reward).
  2026-07-16 seed0 최초 시도(자유 방향 선택, Discrete(4) {Hold,Open-Long,Open-Short,Close})
  에서 "아무것도 안 함" 붕괴 확인됨 (검증 콜백 4회 연속 거래 0건). 원인은 보상 지연이
  아니라 — 이미 매 스텝 밀집 보상이었음 — "진입=확실한 수수료 비용, Hold=항상 0"이라는
  구조적 비대칭. ent_coef 상향/에피소드 길이 조정 모두 무효였음.
  2026-07-20 재설계(완익 사전결정 구조 이슈의 최후 후보안 — V9 Issue -
  완익 사전결정 구조.md): ① 진입 방향은 adaptive/rule과 동일하게 항상 파동 역추세(fade)로
  자동 결정 — 방향 선택이라는 이 프로젝트에서 한 번도 검증된 적 없는 과제를 추가로
  얹지 않고, "보유 중 언제 접을지"라는 원래 풀고자 했던 문제(완익 목표를 진입 시점에
  박제하는 한계)에만 집중하기 위함. 방향 자유 선택 폐기로 Open-Long/Open-Short 구분이
  사라져 Discrete(4)→(3)으로 축소. ② adaptive의 explore_bonus(콜드스타트 함정 돌파용,
  Enter 시 1회 임시 보너스)를 이식 — 06-16 붕괴의 구조적 비대칭에 대한 직접 대응.
  ③ ⚠️ MAX_TRADE_LOSS_PCT 기반 동적 레버리지 상한(adaptive 전용)이 없어, 거래당 최대
  손실이 adaptive의 -20% 예산이 아니라 강제청산가(~-100%)까지 열려있음 — rule 모드와
  동일한 리스크 프로파일이며, 청산 판단을 전적으로 정책의 학습된 판단(관측의 liq_dist
  피처)에 의존한다.

- exit_mode="rule" (V9 Design.md 9장 Fallback B, "진입만 학습"): 정책은 Skip/Enter만
  결정하고(Discrete(2)), 진입 시 방향은 V8과 동일한 파동 역추세(fade) 고정, 청산은
  scratch/backtest_v8.py의 simulate_numba 인-포지션 관리(반익절 tp_half_level,
  본절이동 be_trigger_level, ATR 손절 sl_multiplier)를 그대로 이식한 simulate_v8_exit()가
  전담한다 (스위칭 분기는 제외 — V9.0에서 폐지). Enter 시 청산까지 fast-forward하고
  실현 순손익을 그 스텝에 즉시 보상으로 반환하는 semi-MDP 구조. 매 결정이 완결된 거래
  하나의 결과를 즉시 받으므로 "rl" 모드의 붕괴 원인이던 장기 credit assignment 문제가
  구조적으로 사라진다.

- exit_mode="adaptive" (2026-07-16, rule 모드의 확장): "rule" 모드에서 전역 고정값이던
  sl_multiplier/tp_half_level에 더해 레버리지(1~50 정수)까지 진입 시점에 정책이 직접
  선택하고, 완익절 레벨(tp_full_level)을 신규로 추가하며, 본절이동(BE-move) 로직은 완전히
  제거했다. 증거금은 100 USDT로 고정(V8/rule 모드와 동일 관례, 비교 가능성 유지) —
  "사이즈"가 아니라 "레버리지"를 다이얼로 쓰는 이유는, 둘 다 달러 손익 스케일에는 동일하게
  작용하지만 레버리지만이 청산까지의 거리(=손절이 실제로 유효할 수 있는 폭)를 결정하기
  때문. 행동은 6차원 연속값 Box(-1,1)^6 = [진입여부, 레버리지, 손절폭, 반익절레벨,
  완익절레벨, 반익사용여부] — Enter로 판단되면 나머지 값을 각 파라미터 범위로 매핑해
  simulate_adaptive_exit()에 그대로 넘긴다. "rule" 모드와 마찬가지로 진입 하나 = 결정
  하나 = 완결된 보상인 semi-MDP 구조를 유지하므로(포지션 보유 중 추가 판단 없음) "rl"
  모드의 credit assignment 붕괴를 재도입하지 않는다. 청산은 반익절(50%, tp_half_level,
  a[5]>0일 때만 실행 — 2026-07-19 선택제로 변경) → 완익절(전량, tp_full_level,
  tp_half_level 이상으로 강제) → 고정 손절(sl_multiplier, 진입 후 불변) → 7일
  타임아웃(DEFAULT_MAX_HOLD_BARS) 순으로 판정한다.

- 2026-07-18 (adaptive 모드 보강): 레버리지 상한을 "종목별 고정값"이 아니라 그 순간의
  ATR%에만 반응하는 dynamic_leverage_ceiling()으로 동적 제한한다 (V9 Issue -
  BTC,ETH괴리잡기.md 참고). BTC/ETH에 완전히 동일한 공식을 적용하고 종목 분기는 코드에
  전혀 없음 — ETH가 BTC보다 근접청산이 잦았던 원인(ATR% 구조적으로 더 큼)을, 종목을
  명시하지 않고 "그 순간 변동성이 실제로 높은가"만으로 해소한다. 학습 커리큘럼의
  leverage_max(콜드스타트 분산 억제용)와 이 동적 상한 중 더 타이트한 쪽이 실제 상한이
  되며, 후자는 평가/실전에서도 상시 적용된다.

공통: 시장 메커니즘만 env에 내장 (수수료 0.05%/leg, 거래소 강제청산가는 레버리지·수수료로만
결정되는 가격비율 — 정책이 아무리 넓은 손절을 골라도 이 가격을 넘어설 수 없도록 클램프).
정규화는 고정 상수만 사용 (데이터셋 통계 금지 — 누출 방지, V9 Design.md 3장).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit

MARGIN_USDT = 100.0
OBS_DIM = 18      # 파동/시장 피처 (다이버전스 발생 플래그 포함, wave_age_min 중복 제거).
                  # 2026-07-19: 포지션 상태 4칸 제거 — rule/adaptive는 semi-MDP라 결정 시점에
                  # 항상 무포지션이어서 영원히 0인 죽은 차원이었음. "rl" 모드 전용으로만 유지.
                  # 2026-07-20: 타이밍 피처 4종 추가(14→18) — 현재 RSI(방향조정) + 트레일링
                  # 수익률 3지평(5m/15m/1h, ATR 정규화). 승률 천장 ~27%의 원인이 "되돌림이
                  # 이미 시작됐는지 판정할 정보 부재"(칼손절 즉사 분석)라는 진단에 따름.
RL_OBS_DIM = 22   # "rl" 모드(보유 중 매 스텝 결정) 전용: 파동/시장 18 + 포지션 상태 4
# rl 모드(Discrete(3))와 rule 모드(Discrete(2))가 "0=대기, 1=진입" 관례를 공유 —
# 우연한 값 일치가 아니라 두 모드 다 같은 semi-MDP류 진입 판정 구조라 의도적으로 재사용.
# 2026-07-20: rl 모드 자유 방향 선택(Open-Long/Short 구분, 옛 ACT_LONG/ACT_SHORT/ACT_CLOSE=3)
# 폐기 — 방향은 fade 자동 고정, Discrete(4)→(3)으로 축소.
ACT_HOLD, ACT_ENTER, ACT_CLOSE = 0, 1, 2
ACT_SKIP = ACT_HOLD  # rule 모드 전용 별칭 (가독성)

# 고정 정규화 상수. 분포 리포트(dist_report_v9_*.json)로 BTC/ETH 커버리지 검증 후 확정.
NORM = {
    "wave_scale_max": 12.0,       # % (전체 기간 분포 리포트 기준 BTC/ETH p99 ≈ 10.5~10.7 커버)
    "duration_log_max": np.log1p(10.0),   # 일
    "div_gap_max": 5.0,           # %
    "vol_ratio_log_max": np.log1p(6.0),
    "atr_pct_max": 3.0,           # atr/close %
    "hold_log_max": np.log1p(43200.0),       # 분 (30일, rl 모드 포지션 보유시간 정규화 전용)
    "upnl_clip": (-1.5, 3.0),     # 미실현손익 / 100 USDT
    "liq_dist_max": 10.0,         # %
    # 타이밍 피처: 수익률을 ATR%로 나눈 "몇 ATR만큼 움직였나" 단위의 클립 상한 (2026-07-20).
    # 원시 %는 종목/레짐 간 변동성 차이로 비교 불가라 ATR 정규화 채택 — 조용한 장의 -0.3%와
    # 폭발장의 -0.3%를 구분. 상수는 v9b 분포 리포트(BTC/ETH p1~p99 커버)로 검증 후 확정.
    "ret5_atr_clip": 3.0,
    "ret15_atr_clip": 6.0,
    "ret1h_atr_clip": 12.0,
}

# rule 모드 청산 파라미터 기본값 (V8 Design.md 서치 스페이스 중간값 근사; leverage는 V8과 동일 고정)
DEFAULT_SL_MULTIPLIER = 1.5
DEFAULT_TP_HALF_LEVEL = 0.30
DEFAULT_BE_TRIGGER_LEVEL = 0.60
DEFAULT_MAX_HOLD_BARS = 10080  # 7일 (V8 원본엔 없는, 사전계산 안전을 위한 구현상 상한. 2026-07-16: 30일→7일, 파동 지속시간 분포(BTC p99≈5.4일)에 맞춤)

# adaptive 모드: 정책이 매 진입마다 직접 고르는 4개 파라미터의 매핑 범위
ADAPTIVE_ACTION_DIM = 6  # [진입여부, 레버리지, 손절폭, 반익절레벨, 완익절추가폭, 반익사용여부]
                         # 2026-07-19: 반익 선택제 — a[5]>0이면 반익 실행, 아니면 풀포지션으로
                         # 완익/손절 직행. 강제 반익(자유 없음)과 완전 제거(분산 완충 상실)의 절충.
LEVERAGE_RANGE = (1, 50)            # 정수 레버리지. 증거금은 100 USDT 고정 (V8/rule 모드와 동일 관례).
                                     # 2026-07-17: 100→50 하향. best 체크포인트 실측 결과 근접청산(-95 이하)
                                     # 62건이 전부 레버리지 45배 이상, 89%가 정확히 100배였고 20배 이하는
                                     # 0건 — 상한을 50까지만 열어도 관측된 근접청산 사례를 모두 배제.
                                     # 2026-07-20: 하한 1→3 시도 후 폐기(당일 복귀). 가설(레버리지=1 거래가
                                     # 적자 그룹이니 하한을 올리면 개선)은 인과를 거꾸로 읽은 것이었음 —
                                     # 레버리지=1은 정책의 "확신 약함" 자기평가 신호였고, 하한을 강제로
                                     # 올리자 그 타점들이 확신은 그대로인 채 리스크만 커져 오히려 악화
                                     # (0720-1712 런: 승률은 41~46%로 상승했지만 PnL 전 구간 적자,
                                     # 복리 MDD 96~100%). 레버리지 자기평가 기능을 보존하기 위해 1로 복귀.
SL_MULTIPLIER_RANGE = (0.0, 3.0)    # 2026-07-19: 하한 0.3→0.0. 손절 기준점이 진입가가 아니라 파동
                                     # 극점(end_price)이라 0.0도 "극점 재터치 시 칼손절"이라는 정합적
                                     # 선택임 — 즉사 아님. 나쁜 선택이면 정책이 스스로 회피하게 둔다.
TP_HALF_RANGE = (0.1, 0.5)          # V8과 동일
TP_FULL_EXTRA_RANGE = (0.0, 2.0)    # 완익절레벨 = 반익절레벨 + 이 값 (항상 반익절 이상이 되도록 구성)

# --- 보상 셰이핑 이력 (adaptive 모드) — 현재는 보상 = 실현 손익 그대로 (+explore_bonus 커리큘럼만) ---
# 2026-07-19: +보상 클리핑(REWARD_CLIP_USDT=30) 제거. 50M×2시드 실측 결과 이 전략의 수익 원천인
# "소수의 큰 익절(+100~170)"의 유인을 클립이 정확히 제거해버려서, 정책이 TP를 짧게 당기는
# (승률 26→35%) 대신 수수료+소손실을 못 이기는 음의 기대값으로 적응함. top1 의존 억제는
# 보상이 아니라 모델 선택 기준(eval_v9.v9_score의 top1 제외 항)이 담당한다.
# 2026-07-19: 근접청산(pnl≤-95) 추가 감점(-100)도 제거 — 손실 예산 기반 레버리지 상한
# (MAX_TRADE_LOSS_PCT=20)이 -95 도달 자체를 구조적으로 불가능하게 만들어 죽은 코드가 됨.
# 근접청산 억제는 사후 벌점(보상/점수)이 아니라 사전 차단(상한 공식)으로 일원화.


def _scale(x, lo, hi):
    """Box(-1,1) 값을 [lo, hi] 범위로 선형 매핑"""
    x = min(1.0, max(-1.0, x))
    return lo + (hi - lo) * (x + 1.0) * 0.5


def _scale_int(x, lo, hi):
    """Box(-1,1) 값을 [lo, hi] 정수로 반올림 매핑"""
    val = round(_scale(x, lo, hi))
    return int(min(hi, max(lo, val)))


FEE_MARGIN_PCT = 0.1  # 청산가 공식의 수수료 항(왕복 ~2×0.05%) 보정용 안전 마진 (%p)
MAX_TRADE_LOSS_PCT = 20.0  # 거래 1건의 최대 손실 예산 (증거금 대비 %). 2026-07-19: 상한 공식의
                           # 분자를 100(청산만 회피) → 손실 예산으로 — 갭 항을 추가해 청산
                           # 클램프는 소멸시켰지만, 손절선이 청산가 "바로 안쪽"까지 붙는 건
                           # 여전히 허용돼 정상 손절로도 -95가 나오는 게 실측됨(30K 랜덤 거래
                           # 중 240건). 손절 거리 기준 손실(거리%×lev+수수료)이 예산을 넘지
                           # 않도록 레버리지를 조이는 게 "전재산 청산 금지" 철학에 부합.
                           # 85→20 하향(같은 날): 안전 우선 — 한 거래 최대 -20. 좋은 타점(gap≈0)
                           # + 타이트 손절 조합에서는 여전히 50배까지 열림.


def dynamic_leverage_ceiling(atr_pct, gap_pct=0.0, lev_lo=LEVERAGE_RANGE[0], lev_hi=LEVERAGE_RANGE[1],
                              sl_multiplier=SL_MULTIPLIER_RANGE[1]):
    """
    손절 거리 기반 레버리지 상한 (V9 Issue - BTC,ETH괴리잡기.md 참고).

    핵심 아이디어: "이 거래에서 손절까지 최대로 잃을 수 있는 금액이 증거금의
    MAX_TRADE_LOSS_PCT(%)를 넘지 않도록" 레버리지를 제한한다.

        손절까지의 거리(%) = gap% + ATR% × sl_multiplier
          - gap%  = |진입가 − 파동극점| / 진입가 × 100  (손절선은 파동 극점 기준에 깔리므로)
          - ATR% × sl_multiplier = 극점 바깥으로 두는 버퍼
        최대 손실(%) ≈ 손절거리% × leverage + 수수료
        → leverage_max = MAX_TRADE_LOSS_PCT / (손절거리% + 수수료마진)

    성질:
      - 어떤 (타점, 손절폭, 레버리지) 조합이든 한 거래 손실 ≤ 예산이 수학적으로 보장됨.
        예산 < 95이므로 근접청산(-95)과 강제청산은 구조적으로 불가능.
      - 정책이 실제로 고른 sl_multiplier를 사용(2026-07-19, 최악치 3.0 가정 폐기) —
        타이트한 손절을 고를수록 높은 레버리지가 열리는 "손절 거리 기반 사이징".
        좋은 타점(gap≈0) + 타이트 손절이면 상한 50까지 열린다.
      - BTC/ETH 완전 동일 공식, 종목 분기 없음.

    이력: 초기 공식(2026-07-18)은 ATR 항만 보고 gap 항을 누락해 SL이 청산가로 클램프되는
    -100 경로가 남아있었고(조용한 장일수록 상한이 열리는 역설 포함), gap 항을 추가한 뒤에도
    분자가 100(청산만 회피)이라 정상 손절로 -95가 나오는 게 실측돼 분자를 손실 예산으로
    교체(2026-07-19).
    """
    denom = gap_pct + atr_pct * sl_multiplier + FEE_MARGIN_PCT
    if denom <= 1e-9:
        return float(lev_hi)
    ceiling = MAX_TRADE_LOSS_PCT / denom
    return float(min(lev_hi, max(lev_lo, ceiling)))


@njit(nogil=True, fastmath=True, cache=True)
def simulate_adaptive_exit(
    entry_i, dir_is_bullish, highs, lows, closes,
    start_price, end_price, atr_val,
    leverage, sl_multiplier, tp_half_level, tp_full_level,
    fee_rate, max_hold_bars, last_idx, do_half,
):
    """
    adaptive 모드 청산: 반익절(50%, tp_half_level, do_half=True일 때만) → 완익절(전량,
    tp_full_level) → 고정 손절(진입 후 불변, 본절이동 없음) → 타임아웃 순.
    do_half=False면 반익 없이 풀포지션으로 완익/손절/타임아웃 직행 (2026-07-19 반익 선택제).
    증거금은 100 USDT 고정, leverage는 이 거래 한정으로 정책이 고른 정수값(1~50).
    손절은 이 leverage로 결정되는 거래소 강제청산가를 절대 넘지 못하도록 클램프.
    반환: (exit_idx, net_pnl, exit_price)
    """
    entry_price = closes[entry_i]
    pos_size = (100.0 * leverage) / entry_price
    fee_entry = pos_size * entry_price * fee_rate
    accumulated = -fee_entry

    p_half = start_price * tp_half_level + end_price * (1.0 - tp_half_level)
    p_full = start_price * tp_full_level + end_price * (1.0 - tp_full_level)

    sl_buffer = atr_val * sl_multiplier if atr_val > 0.0 else entry_price * 0.001
    inv_lev = 1.0 / leverage
    if dir_is_bullish:
        liq_price = entry_price * (1.0 + fee_rate - inv_lev) / (1.0 - fee_rate)
        sl_price = max(end_price - sl_buffer, liq_price)
    else:
        liq_price = entry_price * (1.0 - fee_rate + inv_lev) / (1.0 + fee_rate)
        sl_price = min(end_price + sl_buffer, liq_price)

    half_done = not do_half  # 반익 미사용 선택 시 반익 판정을 처음부터 건너뜀
    end_j = min(entry_i + max_hold_bars, last_idx)

    for j in range(entry_i + 1, end_j + 1):
        h = highs[j]
        low = lows[j]
        if dir_is_bullish:
            if h >= p_half and not half_done:
                gross = (p_half - entry_price) * pos_size * 0.5
                fee = pos_size * 0.5 * p_half * fee_rate
                accumulated += gross - fee
                pos_size *= 0.5
                half_done = True
            if h >= p_full:
                gross = (p_full - entry_price) * pos_size
                fee = pos_size * p_full * fee_rate
                accumulated += gross - fee
                return j, accumulated, p_full
            if low <= sl_price:
                gross = (sl_price - entry_price) * pos_size
                fee = pos_size * sl_price * fee_rate
                accumulated += gross - fee
                return j, accumulated, sl_price
        else:
            if low <= p_half and not half_done:
                gross = (entry_price - p_half) * pos_size * 0.5
                fee = pos_size * 0.5 * p_half * fee_rate
                accumulated += gross - fee
                pos_size *= 0.5
                half_done = True
            if low <= p_full:
                gross = (entry_price - p_full) * pos_size
                fee = pos_size * p_full * fee_rate
                accumulated += gross - fee
                return j, accumulated, p_full
            if h >= sl_price:
                gross = (entry_price - sl_price) * pos_size
                fee = pos_size * sl_price * fee_rate
                accumulated += gross - fee
                return j, accumulated, sl_price

    # 타임아웃: max_hold_bars 내 완익/손절 미도달 시 종가 강제 청산
    close_price = closes[end_j]
    if dir_is_bullish:
        gross = (close_price - entry_price) * pos_size
    else:
        gross = (entry_price - close_price) * pos_size
    fee = pos_size * close_price * fee_rate
    accumulated += gross - fee
    return end_j, accumulated, close_price


@njit(nogil=True, fastmath=True, cache=True)
def simulate_v8_exit(
    entry_i, dir_is_bullish, highs, lows, closes,
    start_price, end_price, atr_val,
    sl_multiplier, tp_half_level, be_trigger_level, leverage, fee_rate,
    max_hold_bars, last_idx,
):
    """
    scratch/backtest_v8.py::simulate_numba의 포지션 관리 로직(반익절/본절이동/ATR손절,
    라인 171-201 롱 / 243-272 숏 — 스코어 기반 스위칭 분기 203-241, 274-312는 제외)을
    단일 진입점 시뮬레이션으로 추출한 것. entry_i의 close로 진입해 청산까지 전진한다.
    반환: (exit_idx, net_pnl, exit_price)
    """
    entry_price = closes[entry_i]
    pos_size = (100.0 * leverage) / entry_price
    fee_entry = pos_size * entry_price * fee_rate
    accumulated = -fee_entry

    p_half = start_price * tp_half_level + end_price * (1.0 - tp_half_level)
    p_be = start_price * be_trigger_level + end_price * (1.0 - be_trigger_level)

    sl_buffer = atr_val * sl_multiplier if atr_val > 0.0 else entry_price * 0.001
    if dir_is_bullish:
        max_loss_price = (100.05 * leverage - 20.0) / (pos_size * 0.9995)
        sl_price = max(end_price - sl_buffer, max_loss_price)
    else:
        max_loss_price = (99.95 * leverage + 20.0) / (pos_size * 1.0005)
        sl_price = min(end_price + sl_buffer, max_loss_price)

    half_done = False
    end_j = min(entry_i + max_hold_bars, last_idx)

    for j in range(entry_i + 1, end_j + 1):
        h = highs[j]
        low = lows[j]
        if dir_is_bullish:
            if h >= p_be:
                sl_price = p_half
            if h >= p_half and not half_done:
                gross = (p_half - entry_price) * pos_size * 0.5
                fee = pos_size * 0.5 * p_half * fee_rate
                accumulated += gross - fee
                pos_size *= 0.5
                half_done = True
            if low <= sl_price:
                gross = (sl_price - entry_price) * pos_size
                fee = pos_size * sl_price * fee_rate
                accumulated += gross - fee
                return j, accumulated, sl_price
        else:
            if low <= p_be:
                sl_price = p_half
            if low <= p_half and not half_done:
                gross = (entry_price - p_half) * pos_size * 0.5
                fee = pos_size * 0.5 * p_half * fee_rate
                accumulated += gross - fee
                pos_size *= 0.5
                half_done = True
            if h >= sl_price:
                gross = (entry_price - sl_price) * pos_size
                fee = pos_size * sl_price * fee_rate
                accumulated += gross - fee
                return j, accumulated, sl_price

    # 타임아웃: max_hold_bars 내 SL 미도달 시 종가 강제 청산
    # (V8 원본 simulate_numba엔 없는 항목 — 단일 진입점 사전 시뮬레이션의 계산량 상한을 위한 구현상 경계)
    close_price = closes[end_j]
    if dir_is_bullish:
        gross = (close_price - entry_price) * pos_size
    else:
        gross = (entry_price - close_price) * pos_size
    fee = pos_size * close_price * fee_rate
    accumulated += gross - fee
    return end_j, accumulated, close_price


class TradingEnvV9(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cache_path,
        start_idx=None,
        end_idx=None,
        episode_len_rows=43200,      # 30일 (2026-07-19 60일 확대 → 2026-07-20 복귀: 60일 런 붕괴 실측, train.py 독스트링 참고)
        decision_stride=1,
        leverage=20.0,
        fee_rate=0.0005,
        exit_mode="rule",
        sl_multiplier=DEFAULT_SL_MULTIPLIER,
        tp_half_level=DEFAULT_TP_HALF_LEVEL,
        be_trigger_level=DEFAULT_BE_TRIGGER_LEVEL,
        max_hold_bars=DEFAULT_MAX_HOLD_BARS,
        fixed_full_range=False,      # True면 [start_idx, end_idx)를 단일 에피소드로 (평가용)
        explore_bonus=0.0,           # 학습 커리큘럼 전용 (adaptive 모드 Enter 시 추가 보상, 학습 중반까지 감쇠 후 0).
                                      # 평가(eval_v9.py)는 항상 기본값 0.0으로 새 env를 만들므로 검증/테스트 점수엔 영향 없음.
        leverage_max=LEVERAGE_RANGE[1],  # 학습 커리큘럼 전용 (adaptive 모드 레버리지 상한, 학습 초반엔 낮게 시작해
                                      # 점차 100까지 확대). 레버리지가 높을수록 손절선이 강제청산가에 눌려 거의
                                      # 항상 최대손실(-100)로 청산되기 쉬워, 초반 무작위 탐험의 손실 분산을 줄이는 장치.
                                      # 평가는 항상 기본값(100)이라 최종 판단력엔 영향 없음.
    ):
        super().__init__()
        if exit_mode not in ("rl", "rule", "adaptive"):
            raise NotImplementedError(f"exit_mode='{exit_mode}' not implemented (hybrid는 V9 Design.md 9장 미구현 옵션)")
        self.exit_mode = exit_mode

        data = np.load(cache_path)
        self.cache_path = cache_path
        n = len(data)
        self.lo = 0 if start_idx is None else max(0, int(start_idx))
        self.hi = n if end_idx is None else min(n, int(end_idx))
        if self.hi - self.lo < 100:
            raise ValueError(f"range too small: [{self.lo}, {self.hi})")

        self.highs = data["high"].astype(np.float64)
        self.lows = data["low"].astype(np.float64)
        self.closes = data["close"].astype(np.float64)
        self.ts_ms = data["ts_1m"]
        self.is_bullish_raw = data["is_bullish"].astype(np.bool_)
        self.start_price_raw = data["start_price"].astype(np.float64)
        self.end_price_raw = data["end_price"].astype(np.float64)
        self.atr_5m_raw = data["atr_5m"].astype(np.float64)

        self.static_obs = self._build_static_obs(data)  # (n, 14) float32

        self.episode_len_rows = int(episode_len_rows)
        self.decision_stride = max(1, int(decision_stride))
        self.leverage = float(leverage)
        self.fee_rate = float(fee_rate)
        self.sl_multiplier = float(sl_multiplier)
        self.tp_half_level = float(tp_half_level)
        self.be_trigger_level = float(be_trigger_level)
        self.max_hold_bars = int(max_hold_bars)
        self.fixed_full_range = bool(fixed_full_range)
        self.explore_bonus = float(explore_bonus)  # train_v9.py의 콜백이 학습 중 set_curriculum()으로 갱신
        self.leverage_max = float(leverage_max)    # 위와 동일한 방식으로 갱신

        if exit_mode == "rl":
            self.action_space = spaces.Discrete(3)  # 2026-07-20: {Hold, Enter(fade 자동), Close}
        elif exit_mode == "adaptive":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ADAPTIVE_ACTION_DIM,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(2)
        self._obs_dim = RL_OBS_DIM if exit_mode == "rl" else OBS_DIM
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32)

        self._reset_state()

    def set_curriculum(self, explore_bonus=None, leverage_max=None):
        """학습 커리큘럼 파라미터 갱신 전용. VecEnv.set_attr은 Monitor 등 Wrapper에
        setattr(env_i, ...)를 호출해 래퍼 표면에 그림자 속성을 만들 뿐 내부 env까지
        안 닿는 버그가 있어(SB3 자체 동작 — get_attr은 get_wrapper_attr로 래퍼를 뚫고
        들어가 값이 바뀐 것처럼 착시를 주지만, 실제 step()이 읽는 self.explore_bonus는
        전혀 갱신되지 않았음, 2026-07-17 확인) VecEnv.env_method("set_curriculum", ...)
        로 호출해야 함 — env_method는 get_wrapper_attr로 바운드 메서드를 찾아 호출하므로
        올바르게 내부 env의 self에 적용된다."""
        if explore_bonus is not None:
            self.explore_bonus = float(explore_bonus)
        if leverage_max is not None:
            self.leverage_max = float(leverage_max)

    # ---------- 관측 ----------

    @staticmethod
    def _build_static_obs(data):
        """파동/시장 피처 14차원을 전 행에 대해 사전 정규화 (스텝 시 행 슬라이스만 수행)"""
        n = len(data)
        close = np.maximum(data["close"].astype(np.float64), 1e-9)
        is_bull = data["is_bullish"].astype(np.float64)

        ws = np.clip(data["wave_scale_percent"], 0, NORM["wave_scale_max"]) / NORM["wave_scale_max"]
        dur = np.log1p(np.clip(data["wave_duration_day"], 0, 10.0)) / NORM["duration_log_max"]
        has_div = data["has_rsi_divergence"].astype(np.float64)     # 다이버전스 발생 여부 (0/1)
        rsi = data["rsi_previous"].astype(np.float64)
        # rsi_adj/gap은 has_div=0이면 캐시 단계에서 이미 각각 50.0/0.0(중립값)으로 고정되므로
        # has_div=0 → rsi_adj=0.5, gap=0.0가 항상 성립 — 별도 마스킹 불필요, has_div가 그 신호를 명시.
        rsi_adj = np.where(is_bull > 0, rsi, 100.0 - rsi) / 100.0   # 방향 조정 RSI (simulate_numba :139)
        gap = np.clip(data["divergence_price_gap_percent"], 0, NORM["div_gap_max"]) / NORM["div_gap_max"]
        vr = np.log1p(np.clip(data["volatility_ratio"], 0, 6.0)) / NORM["vol_ratio_log_max"]
        vs = np.clip(data["relative_volume_strength"], 0, 1)
        adx = np.clip(data["adx_5m"], 0, 100) / 100.0
        atr_pct = np.clip(data["atr_5m"] / close * 100.0, 0, NORM["atr_pct_max"]) / NORM["atr_pct_max"]
        bull = is_bull * 2.0 - 1.0                                  # ±1
        fib = np.clip(data["fib_pos"].astype(np.float64), -1.0, 2.0) / 2.0
        prehit = data["pre_hit_0382"].astype(np.float64)
        # wave_age_min은 wave_duration_day와 상관계수 0.87로 사실상 중복이라 관측 피처에서 제외
        # (2026-07-16). 캐시엔 남아있고, 아래 되돌림 델타의 "파동이 그 사이 바뀌었는지" 판정에만 내부적으로 씀.
        age_min = data["wave_age_min"].astype(np.float64)

        # 되돌림 위치 델타 (5m/15m 전 대비). 파동이 그 사이 바뀌었으면 0.
        fib_raw = data["fib_pos"].astype(np.float64)
        d5 = np.zeros(n)
        d15 = np.zeros(n)
        d5[5:] = fib_raw[5:] - fib_raw[:-5]
        d15[15:] = fib_raw[15:] - fib_raw[:-15]
        d5 = np.clip(np.where(age_min >= 5, d5, 0.0), -1.0, 1.0)
        d15 = np.clip(np.where(age_min >= 15, d15, 0.0), -1.0, 1.0)

        # --- 타이밍 피처 4종 (2026-07-20, v9b 캐시 필수) ---
        # 현재 RSI: rsi_adj(다이버전스용 rsi_previous)와 동일한 방향조정 관례 —
        # 파동 방향 기준 "과열 정도"로 통일해 롱/숏 대칭 학습
        rsi_now_adj = np.where(is_bull > 0, data["rsi_now"], 100.0 - data["rsi_now"]) / 100.0
        # 트레일링 수익률: %를 그 시점 ATR%로 나눈 "몇 ATR 움직임"으로 정규화 후 클립
        atr_pct_raw = np.maximum(data["atr_5m"] / close * 100.0, 1e-4)
        r5 = np.clip(data["ret_5m"] * 100.0 / atr_pct_raw, -NORM["ret5_atr_clip"], NORM["ret5_atr_clip"]) / NORM["ret5_atr_clip"]
        r15 = np.clip(data["ret_15m"] * 100.0 / atr_pct_raw, -NORM["ret15_atr_clip"], NORM["ret15_atr_clip"]) / NORM["ret15_atr_clip"]
        r1h = np.clip(data["ret_1h"] * 100.0 / atr_pct_raw, -NORM["ret1h_atr_clip"], NORM["ret1h_atr_clip"]) / NORM["ret1h_atr_clip"]

        return np.stack(
            [ws, dur, has_div, rsi_adj, gap, vr, vs, adx, atr_pct, bull, fib, prehit, d5, d15,
             rsi_now_adj, r5, r15, r1h],
            axis=1,
        ).astype(np.float32)

    def _obs(self):
        i = self._i
        if self.exit_mode in ("rule", "adaptive"):
            # semi-MDP: 결정 시점엔 항상 플랫 → 파동/시장 14차원만 (포지션 상태 칸 제거, 2026-07-19)
            return self.static_obs[i].copy()
        obs = np.empty(RL_OBS_DIM, dtype=np.float32)
        obs[:OBS_DIM] = self.static_obs[i]
        if self.pos_dir == 0:
            obs[OBS_DIM:] = 0.0
        else:
            close = self.closes[i]
            upnl = self._unrealized(close)
            lo_c, hi_c = NORM["upnl_clip"]
            hold_min = (self.ts_ms[i] - self.entry_ts) / 60000.0
            liq_dist = self.pos_dir * (close - self.liq_price) / close * 100.0
            clipped_upnl = np.clip(upnl / MARGIN_USDT, lo_c, hi_c)
            obs[OBS_DIM] = float(self.pos_dir)
            # 2026-07-20: 다른 피처들(대부분 0~1/-1~1)과 스케일을 맞추기 위해 [-1,1]로 재스케일
            # (이전엔 클립 범위(-1.5~3.0) 원시값을 그대로 넣어 이 피처만 유독 큰 값 범위를 가졌음).
            obs[OBS_DIM + 1] = float(2.0 * (clipped_upnl - lo_c) / (hi_c - lo_c) - 1.0)
            obs[OBS_DIM + 2] = float(np.log1p(max(hold_min, 0.0)) / NORM["hold_log_max"])
            obs[OBS_DIM + 3] = float(np.clip(liq_dist, 0.0, NORM["liq_dist_max"]) / NORM["liq_dist_max"])
        return obs

    # ---------- 포지션 회계 (rl 모드) ----------

    def _unrealized(self, price):
        """미실현 순손익: 평가손익 − 현재가 기준 청산 수수료 (청산 시 실현손익과 연속)"""
        gross = self.pos_dir * self.pos_size * (price - self.entry_price)
        return gross - self.pos_size * price * self.fee_rate

    def _equity(self, price):
        eq = self.cum_pnl
        if self.pos_dir != 0:
            eq += self._unrealized(price)
        return eq

    def _open_position(self, direction, i):
        price = self.closes[i]
        self.pos_dir = direction
        self.entry_price = price
        self.pos_size = MARGIN_USDT * self.leverage / price
        self.entry_ts = int(self.ts_ms[i])
        self.cum_pnl -= self.pos_size * price * self.fee_rate  # 진입 수수료 즉시 차감
        f = self.fee_rate
        inv_lev = 1.0 / self.leverage
        if direction > 0:
            # 순손실 = 증거금(100)이 되는 가격 (수수료 포함): 거래소 강제청산 근사
            self.liq_price = price * (1.0 + f - inv_lev) / (1.0 - f)
        else:
            self.liq_price = price * (1.0 - f + inv_lev) / (1.0 + f)

    def _close_position(self, price, ts_ms, reason):
        gross = self.pos_dir * self.pos_size * (price - self.entry_price)
        fees = self.pos_size * (self.entry_price + price) * self.fee_rate
        net = gross - fees
        self.cum_pnl += gross - self.pos_size * price * self.fee_rate  # 진입 수수료는 이미 차감됨
        self.trades.append({
            "entry_ts": self.entry_ts,
            "exit_ts": int(ts_ms),
            "dir": int(self.pos_dir),
            "entry_price": float(self.entry_price),
            "exit_price": float(price),
            "pnl": float(net),
            "reason": reason,
        })
        self.pos_dir = 0
        self.pos_size = 0.0
        self.entry_price = 0.0
        self.liq_price = 0.0

    # ---------- gym API ----------

    def _reset_state(self):
        self._i = self.lo
        self._ep_end = self.hi - 1
        self.pos_dir = 0
        self.pos_size = 0.0
        self.entry_price = 0.0
        self.entry_ts = 0
        self.liq_price = 0.0
        self.cum_pnl = 0.0
        self._equity_prev = 0.0
        self.trades = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.fixed_full_range:
            self._i = self.lo
            self._ep_end = self.hi - 1
        else:
            max_start = self.hi - 1 - self.episode_len_rows
            if max_start <= self.lo:
                self._i = self.lo
                self._ep_end = self.hi - 1
            else:
                self._i = int(self.np_random.integers(self.lo, max_start))
                self._ep_end = self._i + self.episode_len_rows
        return self._obs(), {}

    def step(self, action):
        if self.exit_mode == "rule":
            return self._step_rule(action)
        if self.exit_mode == "adaptive":
            return self._step_adaptive(action)
        return self._step_rl(action)

    # ---------- rule 모드: Skip/Enter, 청산은 V8 엔진 (semi-MDP) ----------

    def _step_rule(self, action):
        i = self._i
        reward = 0.0

        if action == ACT_ENTER:
            wave_is_bullish = bool(self.is_bullish_raw[i])
            dir_is_bullish = not wave_is_bullish  # V8과 동일: 파동 역추세(fade)
            exit_j, net_pnl, exit_price = simulate_v8_exit(
                i, dir_is_bullish, self.highs, self.lows, self.closes,
                self.start_price_raw[i], self.end_price_raw[i], self.atr_5m_raw[i],
                self.sl_multiplier, self.tp_half_level, self.be_trigger_level,
                self.leverage, self.fee_rate, self.max_hold_bars, self.hi - 1,
            )
            self.cum_pnl += net_pnl
            self.trades.append({
                "entry_ts": int(self.ts_ms[i]),
                "exit_ts": int(self.ts_ms[exit_j]),
                "dir": 1 if dir_is_bullish else -1,
                "entry_price": float(self.closes[i]),
                "exit_price": float(exit_price),
                "pnl": float(net_pnl),
                "reason": "v8_exit",
            })
            reward = net_pnl / MARGIN_USDT
            nxt = min(exit_j + 1, self._ep_end)
        else:
            nxt = min(i + self.decision_stride, self._ep_end)

        self._i = nxt
        truncated = nxt >= self._ep_end
        info = {}
        if truncated:
            info["episode_pnl"] = self.cum_pnl
            info["episode_trades"] = len(self.trades)
        return self._obs(), float(reward), False, truncated, info

    # ---------- adaptive 모드: 진입 시 레버리지/손절/반익절/완익절을 정책이 직접 결정 (semi-MDP) ----------

    def _step_adaptive(self, action):
        i = self._i
        reward = 0.0
        a = np.asarray(action, dtype=np.float64).reshape(-1)

        if a[0] > 0.0:  # Enter
            wave_is_bullish = bool(self.is_bullish_raw[i])
            dir_is_bullish = not wave_is_bullish  # V8과 동일: 파동 역추세(fade)

            # self.leverage_max(학습 커리큘럼용, 콜드스타트 분산 억제)와 dynamic_leverage_ceiling
            # (그 순간의 손절 거리로 정해지는 구조적 상한, 항상 적용 — 평가/실전 포함)을 함께 적용.
            # 둘 중 더 타이트한 쪽이 실제 상한이 된다. 후자는 종목 분기 없이 완전히 동일한 공식.
            # 2026-07-19: 정책이 고른 sl_multiplier를 먼저 확정하고 그 값으로 상한을 계산
            # (최악치 3.0 가정 폐기) — "타이트한 손절일수록 높은 레버리지 허용"이라는 손절
            # 거리 기반 사이징. 어떤 조합이든 최대 손실 ≤ MAX_TRADE_LOSS_PCT는 동일하게 보장.
            sl_multiplier = _scale(a[2], *SL_MULTIPLIER_RANGE)
            atr_pct = self.atr_5m_raw[i] / max(self.closes[i], 1e-9) * 100.0
            # gap% = 진입가 ↔ 파동 극점(손절 기준가) 거리 — 2026-07-19: 상한 공식에 갭 항 추가
            gap_pct = abs(self.closes[i] - self.end_price_raw[i]) / max(self.closes[i], 1e-9) * 100.0
            dyn_ceiling = dynamic_leverage_ceiling(atr_pct, gap_pct, sl_multiplier=sl_multiplier)
            effective_leverage_max = min(self.leverage_max, dyn_ceiling)
            leverage_chosen = _scale_int(a[1], LEVERAGE_RANGE[0], effective_leverage_max)  # 증거금은 100 USDT 고정
            tp_half_level = _scale(a[3], *TP_HALF_RANGE)
            tp_full_level = tp_half_level + _scale(a[4], *TP_FULL_EXTRA_RANGE)
            do_half = a[5] > 0.0  # 반익 선택제(2026-07-19): 정책이 반익 실행 여부를 직접 결정

            exit_j, net_pnl, exit_price = simulate_adaptive_exit(
                i, dir_is_bullish, self.highs, self.lows, self.closes,
                self.start_price_raw[i], self.end_price_raw[i], self.atr_5m_raw[i],
                float(leverage_chosen), sl_multiplier, tp_half_level, tp_full_level,
                self.fee_rate, self.max_hold_bars, self.hi - 1, do_half,
            )
            self.cum_pnl += net_pnl
            self.trades.append({
                "entry_ts": int(self.ts_ms[i]),
                "exit_ts": int(self.ts_ms[exit_j]),
                "dir": 1 if dir_is_bullish else -1,
                "entry_price": float(self.closes[i]),
                "exit_price": float(exit_price),
                "pnl": float(net_pnl),
                "reason": "adaptive_exit",
                "leverage": int(leverage_chosen),
                "sl_multiplier": float(sl_multiplier),
                "tp_half_level": float(tp_half_level),
                "tp_full_level": float(tp_full_level),
                "do_half": bool(do_half),
            })
            # 보상 = 실현 손익 그대로 (2026-07-19: 근접청산 추가 감점(-100)도 제거 — 손실 예산
            # 기반 레버리지 상한(MAX_TRADE_LOSS_PCT)이 pnl ≤ -95를 구조적으로 불가능하게 만들어
            # 영원히 발동하지 않는 죽은 코드였음. 사후 벌점 대신 사전 차단으로 대체된 것).
            # explore_bonus는 cum_pnl(실제 손익 집계)엔 더하지 않고 reward에만 적용.
            reward = net_pnl / MARGIN_USDT + self.explore_bonus
            nxt = min(exit_j + 1, self._ep_end)
        else:
            nxt = min(i + self.decision_stride, self._ep_end)

        self._i = nxt
        truncated = nxt >= self._ep_end
        info = {}
        if truncated:
            info["episode_pnl"] = self.cum_pnl
            info["episode_trades"] = len(self.trades)
        return self._obs(), float(reward), False, truncated, info

    # ---------- rl 모드: 보유 중 풀 컨트롤 (dense MtM 보상, 방향은 fade 고정) ----------

    def _step_rl(self, action):
        """2026-07-20 재설계: Discrete(3) {Hold, Enter, Close}. 진입 방향은 adaptive/rule과
        동일하게 항상 파동 역추세(fade)로 자동 결정 — 정책은 진입/청산 "시점"만 학습한다.
        상태에 따라 같은 라벨의 의미가 달라짐(무포지션에서 Close, 보유 중 Enter는 no-op —
        Hold와 동일하게 처리됨)은 설계 논의에서 검토된 정상적인 상태-조건부 정책 동작이며,
        문 앞의 로봇(닫힌 문엔 밀기만 유효, 열린 문엔 당기기만 유효)과 같은 표준 패턴이다.
        방향이 진입 시 고정되므로 옛 버전의 "플립"(반대 방향 즉시 전환) 개념은 사라짐 —
        방향을 바꾸려면 Close 후 다음 결정에서 그 시점 파동의 fade 방향으로 새로 Enter해야
        한다."""
        i = self._i
        ts = self.ts_ms[i]

        # 1) 현재 행 종가에서 행동 적용
        entry_bonus = 0.0
        if action == ACT_ENTER and self.pos_dir == 0:
            wave_is_bullish = bool(self.is_bullish_raw[i])
            dir_is_bullish = not wave_is_bullish  # V8/adaptive/rule과 동일: 파동 역추세(fade)
            self._open_position(1 if dir_is_bullish else -1, i)
            # Enter 시 1회만 부여 (adaptive와 동일 관례) — cum_pnl(실제 손익 집계)엔 미반영,
            # reward에만 적용. 학습 커리큘럼 전용, 평가 env는 항상 explore_bonus=0.0.
            entry_bonus = self.explore_bonus
        elif action == ACT_CLOSE and self.pos_dir != 0:
            self._close_position(self.closes[i], ts, "manual")
        # action==ACT_HOLD, 혹은 상태와 안 맞는 조합(무포지션+Close, 보유중+Enter)은
        # 전부 그대로 통과 — Hold와 동일하게 처리된다.

        # 2) 시간 전진 (stride 구간의 매 행에서 강제청산 체크)
        # 2026-07-21: 목표 길이(self._ep_end)에 도달해도 포지션 보유 중이면 바로 끊지 않고
        # 데이터의 실제 끝(hi-1)까지 연장 — 정책 판단과 무관한 임의 시점 강제청산이 학습
        # 신호에 노이즈를 더하는 것을 방지 (무포지션 상태에서만 truncate). 데이터가 정말
        # 끝나버리면(hi-1 도달) 그때는 더 진행할 데이터가 없으므로 부득이 종가 정산한다.
        hard_end = self.hi - 1
        nxt = min(i + self.decision_stride, hard_end)
        for j in range(i + 1, nxt + 1):
            if self.pos_dir != 0:
                if self.pos_dir > 0 and self.lows[j] <= self.liq_price:
                    self._close_position(self.liq_price, self.ts_ms[j], "liquidation")
                elif self.pos_dir < 0 and self.highs[j] >= self.liq_price:
                    self._close_position(self.liq_price, self.ts_ms[j], "liquidation")
        self._i = nxt

        # 3) 종료 판정: 데이터 끝에 도달했거나(무조건), 목표 길이에 도달 + 무포지션일 때만
        at_hard_end = nxt >= hard_end
        truncated = at_hard_end or (nxt >= self._ep_end and self.pos_dir == 0)
        if truncated and self.pos_dir != 0:
            self._close_position(self.closes[nxt], self.ts_ms[nxt], "episode_end")

        # 4) 보상 = MtM 자산 변화량 / 100 USDT + Enter 시 1회 탐험 보너스
        equity = self._equity(self.closes[nxt])
        reward = (equity - self._equity_prev) / MARGIN_USDT + entry_bonus
        self._equity_prev = equity

        info = {}
        if truncated:
            info["episode_pnl"] = self.cum_pnl
            info["episode_trades"] = len(self.trades)
        return self._obs(), float(reward), False, truncated, info
