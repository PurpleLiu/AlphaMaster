"""
prune_features.py — 特徵剪枝：從 65 個特徵中篩出有區分度、低冗餘的子集

背景：
    特徵庫擴展到 65 個後，vocab=131，8-token 搜索空間暴增到 ~8.67×10^16。
    其中大量特徵高度相關（多週期均線/動量/通道位置類），是噪聲維度。

做法（用 model_core.evaluator）：
    1. 離線載入全部品種數據，直接從 _FEATURE_DEFS 計算全部 65 個特徵
    2. score_all：對每個特徵算 IC / RankIC / 互資訊，跨特徵秩歸一聚合 importance
    3. prune：保守雙條件相關性剪枝（corr>閾值 且 分差>margin 才剪）
    4. 額外按 importance 取 Top-K，去掉近零 IC 的弱特徵
    5. 寫出 active_features.json（features.py 啟動時讀取，只註冊白名單特徵）

用法：
    python prune_features.py                 # 默認 corr_threshold=0.85, top_k=28
    python prune_features.py --top-k 25      # 自訂保留數量
    python prune_features.py --corr 0.9      # 自訂相關性閾值
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.features import _FEATURE_DEFS
from model_core.evaluator import score_all, prune

OUTPUT = Path(__file__).parent / "active_features.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=28,
                    help="剪枝後額外按 importance 截斷保留的最大特徵數")
    ap.add_argument("--corr", type=float, default=0.85,
                    help="相關性剪枝閾值（越低剪得越狠）")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="剪枝分差閾值（勢均力敵時保留）")
    args = ap.parse_args()

    print(f"{'='*62}")
    print(f"  特徵剪枝  corr_threshold={args.corr}  top_k={args.top_k}")
    print(f"{'='*62}")

    # ── 1. 離線載入數據 ───────────────────────────────────────────────
    print("載入數據（離線快取）...")
    with MT5DataFetcher(offline=True) as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        raw_dict = mgr.raw_dict
        target   = mgr.target_ret          # [N, T]
        syms     = mgr.symbols
    N, T = target.shape
    print(f"  品種={syms}  N={N}  T={T}")

    # ── 2. 直接從 _FEATURE_DEFS 計算全部 65 個特徵（繞過白名單）───────
    print(f"\n計算全部 {len(_FEATURE_DEFS)} 個特徵...")
    candidates: dict[str, torch.Tensor] = {}
    categories: dict[str, str] = {}
    for name, category, compute in _FEATURE_DEFS:
        try:
            series = compute(raw_dict)                 # [N, T]
            candidates[name] = torch.nan_to_num(series, nan=0.0,
                                                posinf=0.0, neginf=0.0)
            categories[name] = category
        except Exception as e:
            print(f"  [跳過] {name}: 計算失敗 {e}")

    # ── 3. 打分 ───────────────────────────────────────────────────────
    print(f"\n對 {len(candidates)} 個特徵打分（IC / RankIC / MI）...")
    scores = score_all(candidates, target, categories=categories)
    scores_sorted = sorted(scores, key=lambda s: (
        -s.importance_score if s.importance_score == s.importance_score else 1,
    ))

    print(f"\n  {'特徵':22s}{'類別':14s}{'IC':>8}{'RankIC':>8}{'MI':>8}{'importance':>12}")
    print(f"  {'-'*70}")
    for s in scores_sorted:
        imp = s.importance_score
        imp_str = f"{imp:.4f}" if imp == imp and imp > -1e30 else "  退化"
        print(f"  {s.candidate:22s}{s.category:14s}"
              f"{s.ic:+8.4f}{s.rank_ic:+8.4f}{s.mi:8.4f}{imp_str:>12}")

    # ── 4. 相關性剪枝 ─────────────────────────────────────────────────
    print(f"\n相關性剪枝（corr>{args.corr} 且分差>{args.margin}）...")
    rows = prune(scores, candidates, corr_threshold=args.corr, margin=args.margin)
    retained = [r for r in rows if r.retention_status == "retained"]
    pruned   = [r for r in rows if r.retention_status == "pruned"]
    print(f"  相關性剪枝後保留 {len(retained)} 個，剪掉 {len(pruned)} 個")
    for r in pruned:
        print(f"    ✗ {r.candidate:22s} (favor of {r.pruned_in_favor_of})")

    # ── 5. 按 importance 取 Top-K（去掉弱特徵）────────────────────────
    retained_sorted = sorted(
        retained,
        key=lambda r: (r.importance_score if r.importance_score == r.importance_score
                       and r.importance_score > -1e30 else -1e30),
        reverse=True,
    )
    final = retained_sorted[:args.top_k]
    final_names_set = {r.candidate for r in final}

    # 保持 _FEATURE_DEFS 原始順序輸出
    ordered_names = [name for name, _, _ in _FEATURE_DEFS if name in final_names_set]

    print(f"\n{'='*62}")
    print(f"  最終保留 {len(ordered_names)} 個特徵（vocab 從 65 特徵降至 {len(ordered_names)}）")
    print(f"{'='*62}")
    for n in ordered_names:
        print(f"  ✓ {n}")

    # ── 6. 寫出 active_features.json ─────────────────────────────────
    payload = {
        "active_features": ordered_names,
        "meta": {
            "source": "prune_features.py",
            "corr_threshold": args.corr,
            "top_k": args.top_k,
            "n_original": len(_FEATURE_DEFS),
            "n_retained": len(ordered_names),
            "symbols": syms,
            "bars": T,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\n已寫出 → {OUTPUT}")
    print("重新 import model_core.features 時將只註冊白名單特徵。\n")


if __name__ == "__main__":
    main()
