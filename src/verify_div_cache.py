"""
RSI 다이버전스 피처 검증 게이트 (2026-08-02, RSI-div-개편이전-핵심결함.md Step 4)

피처 캐시의 다이버전스가 "되돌림 진행도 측정기"가 아니라 진짜 다이버전스인지 판정한다.
v9b(개편 전)와 v9e(개편 후) 캐시 모두에서 동작 — v9b는 결함이 재현되어야 하고(FAIL),
v9e는 아래 게이트를 전부 통과해야 한다(PASS):

  G1. 에피소드 내 갭 표류율 < 5% — 같은 극점 쌍이 유지되는 동안(연속 div=1 & end_price/
      rsi_previous 동일) 갭이 변하면 안 됨. 구 결함(rsi2를 '현재 봉'에서 읽음)은 갭이 매 5m마다
      표류해 에피소드의 38%+에서 검출된다. 0이 아닌 이유: 가격이 이전 극점을 f4 표시로는 동일,
      f8로는 근소하게 넘는 재터치(더블탑/바텀)는 정당한 극점 교체인데 f4 키로는 같은 에피소드로
      보임 — 실측 <1%라 5% 게이트로 양쪽이 명확히 분리됨.
      ※ 초기 게이트였던 행 풀링 |corr(fib_pos, 갭)|<0.1은 폐기 (2026-08-02): 에피소드 단위
      분석 결과 BTC/ETH/SOL 모두 극점 쌍 확정 시점 기준 corr +0.26~+0.32로, 이는 "갭이 큰
      다이버전스일수록 이후 되돌림이 깊다"는 시장 신호(모델이 학습할 대상)와 에피소드 길이
      구성 효과이지 누출이 아님. 행 풀링 corr은 참고 지표로만 출력.
  G2. 부호모순(has_div=1인데 갭<=0) 0건        — (가격,RSI) 짝이 같은 봉에서 읽혔다는 증거
  G3. has_div=1이면 갭 >= DIV_MIN_RSI_GAP      — 최소 갭 필터 동작 확인

사용법:
  python src/verify_div_cache.py caches/features_v9e_BTC-USDT-SWAP.npy [...]
  python src/verify_div_cache.py --symbols BTC-USDT-SWAP ETH-USDT-SWAP SOL-USDT-SWAP
"""
import os
import sys
import argparse

import numpy as np

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

from prep_features import CACHE_DIR, CACHE_VER, DIV_MIN_RSI_GAP  # noqa: E402


def episode_gap_drift(d, gap):
    """div=1 행들을 에피소드(연속 i_1m & 동일 end_price/rsi_previous)로 묶어
    에피소드 내 갭 변동(max-min)의 최댓값과 위반 에피소드 수를 반환."""
    key = np.stack([d["end_price"].astype(np.float64), d["rsi_previous"].astype(np.float64)], axis=1)
    new_ep = np.ones(len(d), dtype=bool)
    new_ep[1:] = (np.diff(d["i_1m"]) != 1) | (key[1:] != key[:-1]).any(axis=1)
    ep_id = np.cumsum(new_ep) - 1
    n_ep = int(ep_id[-1]) + 1
    # 에피소드별 min/max를 벡터화로 산출
    order = np.argsort(ep_id, kind="stable")
    g = gap[order]
    e = ep_id[order]
    starts = np.searchsorted(e, np.arange(n_ep))
    ends = np.append(starts[1:], len(g))
    drift = np.array([g[s:t].max() - g[s:t].min() for s, t in zip(starts, ends)])
    return n_ep, int((drift > 1e-6).sum()), float(drift.max())


def verify(path):
    a = np.load(path, mmap_mode="r")
    n = len(a)
    has_gap_field = "rsi_divergence_gap" in a.dtype.names

    div = a["has_rsi_divergence"] == 1
    n_div = int(div.sum())
    name = os.path.basename(path)
    if n_div == 0:
        print(f"[{name}] rows={n:,}  div=1: 0건 — 판정 불가(캐시 생성 로직 확인 필요)")
        return False

    d = a[div]  # div=1 행만 메모리로 실체화 (에피소드 분석용)
    is_bull = d["is_bullish"] == 1
    fib = d["fib_pos"].astype(np.float64)
    age = d["wave_age_min"].astype(np.float64)
    rsi1 = d["rsi_previous"].astype(np.float64)

    if has_gap_field:
        gap = d["rsi_divergence_gap"].astype(np.float64)
        gap_src = "rsi_divergence_gap(저장값)"
    else:
        # v9b 캐시: 갭 미저장 — 구 결함 재현용 근사(rsi2 ≈ rsi_now, 개편 전엔 사실상 항등)
        rsi_now = d["rsi_now"].astype(np.float64)
        gap = np.where(is_bull, rsi1 - rsi_now, rsi_now - rsi1)
        gap_src = "rsi_previous - rsi_now(v9b 근사)"

    n_ep, n_drift_ep, max_drift = episode_gap_drift(d, gap)
    corr = float(np.corrcoef(fib, gap)[0, 1])
    n_contradiction = int((gap <= 0).sum())
    n_below_min = int((gap < DIV_MIN_RSI_GAP).sum())

    drift_ratio = n_drift_ep / n_ep
    g1 = drift_ratio < 0.05
    g2 = n_contradiction == 0
    g3 = n_below_min == 0
    ok = g1 and g2 and g3

    print(f"[{name}] rows={n:,}  div=1: {n_div:,} ({n_div / n * 100:.2f}%)  갭 출처: {gap_src}")
    print(f"  G1 에피소드 내 갭 표류 = {n_drift_ep:,}/{n_ep:,} ({drift_ratio * 100:.1f}%, 게이트<5%, 최대 {max_drift:.1f}pt) → {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 부호모순(갭<=0)     = {n_contradiction:,}건                           → {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 갭<{DIV_MIN_RSI_GAP}pt            = {n_below_min:,}건                           → {'PASS' if g3 else 'FAIL'}")
    print(f"  참고: 행 풀링 corr(fib_pos, 갭) {corr:+.3f} (신호+구성 효과, 게이트 아님), "
          f"갭 중앙값 {np.median(gap):.1f}pt / p95 {np.percentile(gap, 95):.1f}pt")
    print(f"  참고: div행 wave_age 중앙값 {np.median(age):.0f}분, "
          f"bearish 중 rsi1>70 {float((rsi1[is_bull] > 70).mean()) * 100:.1f}%")
    print(f"  ==> {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="검증할 캐시 .npy 경로들")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="심볼명으로 현재 CACHE_VER 캐시를 검증 (paths 대신)")
    parser.add_argument("--suffix", default="", help="캐시 suffix (예: _recent120d)")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.symbols:
        paths += [os.path.join(CACHE_DIR, f"features_{CACHE_VER}_{s}{args.suffix}.npy") for s in args.symbols]
    if not paths:
        parser.error("캐시 경로 또는 --symbols 필요")

    results = [verify(p) for p in paths]
    if all(results):
        print("전체 PASS")
    else:
        print("FAIL 존재 — 위 게이트 위반 항목 확인")
        sys.exit(1)


if __name__ == "__main__":
    main()
