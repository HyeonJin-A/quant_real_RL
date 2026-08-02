"""
V9 트레이딩 환경 (V9 Design.md 5~6장, 9장)

exit_mode="rl" (보유 중 풀 컨트롤) 단일 모드: 결정 주기 매 1m 행, 행동 Discrete(3)
{0: Hold, 1: Enter, 2: Close}. 보상은 매 스텝 mark-to-market 손익 변화량(dense reward).
2026-07-16 seed0 최초 시도(자유 방향 선택, Discrete(4) {Hold,Open-Long,Open-Short,Close})
에서 "아무것도 안 함" 붕괴 확인됨 (검증 콜백 4회 연속 거래 0건). 원인은 보상 지연이
아니라 — 이미 매 스텝 밀집 보상이었음 — "진입=확실한 수수료 비용, Hold=항상 0"이라는
구조적 비대칭. ent_coef 상향/에피소드 길이 조정 모두 무효였음.
2026-07-20 재설계(완익 사전결정 구조 이슈의 최후 후보안 — V9 Issue -
완익 사전결정 구조.md): ① 진입 방향은 항상 파동 역추세(fade)로 자동 결정 — 방향 선택이라는
이 프로젝트에서 한 번도 검증된 적 없는 과제를 추가로 얹지 않고, "보유 중 언제 접을지"라는
원래 풀고자 했던 문제(완익 목표를 진입 시점에 박제하는 한계)에만 집중하기 위함. 방향 자유
선택 폐기로 Open-Long/Open-Short 구분이 사라져 Discrete(4)→(3)으로 축소. ② explore_bonus
(콜드스타트 함정 돌파용, Enter 시 1회 임시 보너스) 도입 — 06-16 붕괴의 구조적 비대칭에 대한
직접 대응. ③ ⚠️ 동적 레버리지 상한이 없어, 거래당 최대 손실이 강제청산가(~-100%)까지
열려있음 — 청산 판단을 전적으로 정책의 학습된 판단(관측의 liq_dist 피처)에 의존한다.

2026-07-30: exit_mode="rule"/"adaptive"(레거시 semi-MDP 모드) 및 관련 구현
(simulate_v8_exit/simulate_adaptive_exit/dynamic_leverage_ceiling 등) 완전 제거 — "rl"이
유일한 지원 모드.

시장 메커니즘만 env에 내장 (수수료 0.05%/leg, 거래소 강제청산가는 레버리지·수수료로만
결정되는 가격비율 — 정책이 아무리 넓은 손절을 골라도 이 가격을 넘어설 수 없도록 클램프).
정규화는 고정 상수만 사용 (데이터셋 통계 금지 — 누출 방지, V9 Design.md 3장).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

MARGIN_USDT = 100.0
OBS_DIM = 18      # 파동/시장 피처 (다이버전스 발생 플래그 포함, wave_age_min 중복 제거).
                  # 2026-07-19: 포지션 상태 4칸 제거 — 옛 rule/adaptive 모드는 semi-MDP라 결정 시점에
                  # 항상 무포지션이어서 영원히 0인 죽은 차원이었음. "rl" 모드 전용으로만 유지.
                  # 2026-07-20: 타이밍 피처 4종 추가(14→18) — 현재 RSI(방향조정) + 트레일링
                  # 수익률 3지평(5m/15m/1h, ATR 정규화). 승률 천장 ~27%의 원인이 "되돌림이
                  # 이미 시작됐는지 판정할 정보 부재"(칼손절 즉사 분석)라는 진단에 따름.
RL_OBS_DIM = 22   # "rl" 모드(보유 중 매 스텝 결정) 전용: 파동/시장 18 + 포지션 상태 4
# 2026-07-20: rl 모드 자유 방향 선택(Open-Long/Short 구분, 옛 ACT_LONG/ACT_SHORT/ACT_CLOSE=3)
# 폐기 — 방향은 fade 자동 고정, Discrete(4)→(3)으로 축소.
ACT_HOLD, ACT_ENTER, ACT_CLOSE = 0, 1, 2

# 고정 정규화 상수. 분포 리포트(dist_report_v9_*.json)로 BTC/ETH 커버리지 검증 후 확정.
NORM = {
    "wave_scale_max": 12.0,       # % (전체 기간 분포 리포트 기준 BTC/ETH p99 ≈ 10.5~10.7 커버)
    "duration_log_max": np.log1p(10.0),   # 일
    "div_gap_max": 5.0,           # %
    "vol_ratio_log_max": np.log1p(6.0),
    "atr_pct_max": 3.0,           # atr/close %
    "hold_log_max": np.log1p(43200.0),       # 분 (30일, rl 모드 포지션 보유시간 정규화 전용)
    "upnl_clip": (-1.5, 3.0),     # 미실현손익 / 100 USDT
    # 타이밍 피처: 수익률을 ATR%로 나눈 "몇 ATR만큼 움직였나" 단위의 클립 상한 (2026-07-20).
    # 원시 %는 종목/레짐 간 변동성 차이로 비교 불가라 ATR 정규화 채택 — 조용한 장의 -0.3%와
    # 폭발장의 -0.3%를 구분. 상수는 v9b 분포 리포트(BTC/ETH p1~p99 커버)로 검증 후 확정.
    "ret5_atr_clip": 3.0,
    "ret15_atr_clip": 6.0,
    "ret1h_atr_clip": 12.0,
}

DEFAULT_MAX_HOLD_BARS = 10080  # 7일 (사전계산 안전을 위한 구현상 상한. 2026-07-16: 30일→7일, 파동 지속시간 분포(BTC p99≈5.4일)에 맞춤)


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
        fee_rate=0.0007,
        max_hold_bars=DEFAULT_MAX_HOLD_BARS,
        fixed_full_range=False,      # True면 [start_idx, end_idx)를 단일 에피소드로 (평가용)
        explore_bonus=0.0,           # 학습 커리큘럼 전용 (Enter 시 추가 보상, 학습 중반까지 감쇠 후 0).
                                      # 평가(eval_v9.py)는 항상 기본값 0.0으로 새 env를 만들므로 검증/테스트 점수엔 영향 없음.
    ):
        super().__init__()

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
        self.max_hold_bars = int(max_hold_bars)
        self.fixed_full_range = bool(fixed_full_range)
        self.explore_bonus = float(explore_bonus)  # train_v9.py의 콜백이 학습 중 set_curriculum()으로 갱신

        self.action_space = spaces.Discrete(3)  # 2026-07-20: {Hold, Enter(fade 자동), Close}
        self._obs_dim = RL_OBS_DIM
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32)

        self._reset_state()

    def set_curriculum(self, explore_bonus=None):
        """학습 커리큘럼 파라미터 갱신 전용. VecEnv.set_attr은 Monitor 등 Wrapper에
        setattr(env_i, ...)를 호출해 래퍼 표면에 그림자 속성을 만들 뿐 내부 env까지
        안 닿는 버그가 있어(SB3 자체 동작 — get_attr은 get_wrapper_attr로 래퍼를 뚫고
        들어가 값이 바뀐 것처럼 착시를 주지만, 실제 step()이 읽는 self.explore_bonus는
        전혀 갱신되지 않았음, 2026-07-17 확인) VecEnv.env_method("set_curriculum", ...)
        로 호출해야 함 — env_method는 get_wrapper_attr로 바운드 메서드를 찾아 호출하므로
        올바르게 내부 env의 self에 적용된다."""
        if explore_bonus is not None:
            self.explore_bonus = float(explore_bonus)

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
            # 2026-07-24: 하드 클립(구 liq_dist_max=10%) 제거 — leverage에 따라 진입 시점부터
            # liq_dist(~100/leverage%)가 이미 옛 상한을 넘어, 청산 근접 정보가 보유기간 대부분
            # 1.0으로 뭉개져 있었음. log1p 압축으로 교체: 위험(0 근접) 구간엔 해상도를 몰아주고
            # 안전 구간은 압축하되 정보를 완전히 버리진 않음 — 기준값은 진입 시점 값(100/leverage)
            # 이라 obs=1.0이 "진입 때만큼 안전", <1이면 그보다 위험, >1이면 그보다 안전을 의미.
            liq_dist_ref = 100.0 / self.leverage
            obs[OBS_DIM + 3] = float(np.log1p(max(liq_dist, 0.0)) / np.log1p(liq_dist_ref))
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
        self._trade_peak_upnl = 0.0  # 2026-07-25: upnl_clip 상한 재검토용 참고 지표 (거래 중 관측 최대 upnl/증거금 배수)
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
        # 청산가 자체도 후보에 포함 (강제청산/마지막 홀드 봉에서만 갱신되던 값이 아니라
        # 실제 청산 체결가 기준 upnl까지 반영)
        peak_upnl = max(self._trade_peak_upnl, self._unrealized(price) / MARGIN_USDT)
        self.trades.append({
            "entry_ts": self.entry_ts,
            "exit_ts": int(ts_ms),
            "dir": int(self.pos_dir),
            "entry_price": float(self.entry_price),
            "exit_price": float(price),
            "pnl": float(net),
            "reason": reason,
            "max_upnl": float(peak_upnl),
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
        self._trade_peak_upnl = 0.0
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
        return self._step_rl(action)

    # ---------- rl 모드: 보유 중 풀 컨트롤 (dense MtM 보상, 방향은 fade 고정) ----------

    def action_masks(self):
        """2026-07-27: sb3-contrib MaskablePPO용 — 상태에 안 맞는 행동(무포지션+Close,
        보유중+Enter)을 후보에서 물리적으로 제거. {Hold, Enter, Close} 순서
        (ACT_HOLD/ACT_ENTER/ACT_CLOSE와 동일 인덱스)."""
        if self.pos_dir == 0:
            return [True, True, False]
        return [True, False, True]

    def _step_rl(self, action):
        """2026-07-20 재설계: Discrete(3) {Hold, Enter, Close}. 진입 방향은 V8/옛 adaptive/rule
        모드와 동일하게 항상 파동 역추세(fade)로 자동 결정 — 정책은 진입/청산 "시점"만 학습한다.
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
            dir_is_bullish = not wave_is_bullish  # V8과 동일: 파동 역추세(fade)
            self._open_position(1 if dir_is_bullish else -1, i)
            # Enter 시 1회만 부여 — cum_pnl(실제 손익 집계)엔 미반영,
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
                self._trade_peak_upnl = max(
                    self._trade_peak_upnl, self._unrealized(self.closes[j]) / MARGIN_USDT
                )
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
