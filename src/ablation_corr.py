"""관측 정적 피처(FEATURE_NAMES, 기본 18종) 쌍별 상관계수 분석 — 캐시 데이터만 사용, 학습 불필요.

피처 후진제거(backward elimination) 실험의 후보 우선순위 큐를 산출한다
(`V10 Design TODO - 관측 피처 후진제거 (Feature Ablation).md` 참고). 이미 확립된 전례:
wave_age_min은 wave_duration_day와 상관계수 0.87로 중복이라 관측에서 제외됐다(2026-07-16,
env.py FEATURE_NAMES 주석 참고) — 이 수치를 "확실히 먼저 테스트해볼 만하다"는 대략적 기준선으로 삼는다.

TradingEnvV9._build_static_obs를 staticmethod 그대로 직접 호출해(env 인스턴스 생성 불필요)
정규화 후 값(실제 정책 입력 공간) 기준으로 상관계수를 계산한다. train split(누출 방지 원칙,
NORM 주석 참고)만 사용 — valid/test는 건드리지 않는다.

사용법:
  python src/ablation_corr.py --symbols BTC-USDT-SWAP ETH-USDT-SWAP --cache-ver v10
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import TradingEnvV9, FEATURE_NAMES  # noqa: E402
from eval import cache_path_for, split_bounds, CACHE_DIR, CACHE_VER  # noqa: E402

# wave_age_min vs wave_duration_day 상관 0.87로 제외된 전례 (2026-07-16) — 참고 기준선.
KNOWN_REDUNDANT_CORR = 0.87


def corr_for_symbol(cache_path, lo, hi):
    data = np.load(cache_path)
    static_obs, names = TradingEnvV9._build_static_obs(data)  # exclude_features 없이 전체 계산
    assert names == list(FEATURE_NAMES)
    x = static_obs[lo:hi]
    corr = np.corrcoef(x, rowvar=False)
    return corr, names


def rank_features(corr, names):
    """피처별 max_j≠i |corr(i,j)| 내림차순 — 이 순서가 그대로 제거 후보 테스트 큐가 된다."""
    n = len(names)
    max_abs = np.zeros(n)
    partner = [None] * n
    for i in range(n):
        row = np.abs(corr[i]).copy()
        row[i] = -1.0  # 자기 자신 제외
        j = int(np.argmax(row))
        max_abs[i] = row[j]
        partner[i] = names[j]
    order = np.argsort(-max_abs)
    return [
        {"feature": names[i], "max_abs_corr": float(max_abs[i]), "with": partner[i]}
        for i in order
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT-SWAP", "ETH-USDT-SWAP"])
    parser.add_argument("--cache-ver", default=None,
                        help="캐시 스키마 버전. 미지정 시 eval.CACHE_VER(최신) 사용")
    parser.add_argument("--cache-suffix", default="", help="스모크용 캐시 suffix (예: _recent120d)")
    parser.add_argument("--final-split", action="store_true",
                        help="배포 전 최종학습용 95/5(train/valid) 분할 기준 (기본은 개발용 70/15/15)")
    args = parser.parse_args()

    cache_ver = args.cache_ver or CACHE_VER
    paths = {s: cache_path_for(s, args.cache_suffix, cache_ver=cache_ver) for s in args.symbols}
    for s, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"cache not found for {s}: {p}")
    # train.py/eval.py와 동일하게 전 심볼 공통 날짜 경계로 분할 (심볼별 독립 분할과 결과가 달라짐)
    bounds = split_bounds(list(paths.values()), final=args.final_split)

    per_symbol_ranking = {}
    for sym, path in paths.items():
        lo, hi = bounds[path]["train"]
        corr, names = corr_for_symbol(path, lo, hi)
        ranking = rank_features(corr, names)
        per_symbol_ranking[sym] = ranking

        print(f"\n=== {sym} (train rows [{lo}:{hi})) ===")
        for r in ranking:
            flag = "  <-- 0.87(wave_age_min 전례) 이상, 우선 테스트 권장" \
                if r["max_abs_corr"] >= KNOWN_REDUNDANT_CORR else ""
            print(f"  {r['feature']:<14} max|corr|={r['max_abs_corr']:.3f}  (with {r['with']}){flag}")

        out_path = os.path.join(CACHE_DIR, f"ablation_corr_{cache_ver}_{sym}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "symbol": sym, "cache_path": path, "cache_ver": cache_ver,
                "train_range": [lo, hi], "feature_names": names,
                "corr_matrix": corr.tolist(), "ranking": ranking,
            }, f, indent=2)
        print(f"saved: {out_path}")

    # 심볼 간 교차확인 — 두 심볼 모두에서 상위권이면 더 신뢰할 만한 후보 (모델 선택은 BTC 우선이지만
    # ETH divergence는 100M 스텝 실험을 걸기 전 human sanity check 용도로 참고).
    if len(per_symbol_ranking) > 1:
        print("\n=== 심볼 간 교차확인 (평균 max|corr| 내림차순) ===")
        by_feature = {name: [] for name in FEATURE_NAMES}
        for ranking in per_symbol_ranking.values():
            for r in ranking:
                by_feature[r["feature"]].append(r["max_abs_corr"])
        avg_ranking = sorted(
            ((name, float(np.mean(vals))) for name, vals in by_feature.items()),
            key=lambda kv: -kv[1],
        )
        for name, avg in avg_ranking:
            print(f"  {name:<14} avg max|corr|={avg:.3f}")


if __name__ == "__main__":
    main()
