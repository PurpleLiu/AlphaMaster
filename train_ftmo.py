"""
train_ftmo.py — FTMO 專屬訓練（跳過 forex，訓練 metals_comm + index）

獎勵函數已切換為 FTMO 模式（ModelConfig.REWARD_MODE='ftmo'）：
  - 年化收益權重 0.60 → 0.75（提權 25%）
  - Calmar 權重 0.05 → 0.10（控制 MDD 貼近 10% Max Loss）
  - IC/Sortino/一致性等降權

用法：
    python train_ftmo.py --offline
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from main import train_group

# 跳過已完成的 forex 組，訓練其餘組
GROUPS_TO_TRAIN = ["metals_comm", "index"]


def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    # 確認 FTMO 模式已啟用
    print(f"\n{'='*60}")
    print(f"  FTMO 專屬因子訓練")
    print(f"{'='*60}")
    print(f"  獎勵模式: REWARD_MODE = '{ModelConfig.REWARD_MODE}'")
    if ModelConfig.REWARD_MODE != "ftmo":
        print(f"  [警告] 當前不是 ftmo 模式，請檢查 model_core/config.py")
        return
    print(f"  權重: ann_ret=0.75  calmar=0.10  sortino=0.05")
    print(f"  跳過: forex（已完成，Best=0.485，standard 模式）")
    print(f"  訓練: {GROUPS_TO_TRAIN}")
    print(f"  offline={offline}  BARS_COUNT={Config.BARS_COUNT}")
    print(f"  TRAIN_STEPS={ModelConfig.TRAIN_STEPS}  BATCH_SIZE={ModelConfig.BATCH_SIZE}")
    print(f"{'='*60}")

    results = {}
    with MT5DataFetcher(offline=offline) as fetcher:
        for gname in GROUPS_TO_TRAIN:
            gsyms = Config.SYMBOL_GROUPS.get(gname, [])
            if not gsyms:
                print(f"  [跳過] 組 {gname} 不存在")
                continue

            print(f"\n>>> 開始訓練 [{gname}] 組: {gsyms}")
            eng = train_group(fetcher, gname, gsyms, offline)
            if eng is not None:
                results[gname] = {
                    "score": eng.best_score,
                    "formula": eng.best_formula,
                    "readable": eng._decode_formula(eng.best_formula),
                }
                print(f"<<< [{gname}] 完成: score={eng.best_score:.4f}")
                print(f"    {results[gname]['readable']}")
            else:
                print(f"<<< [{gname}] 失敗")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  FTMO 訓練完成  耗時 {elapsed/3600:.2f}h")
    print(f"{'='*60}")
    for gname, r in results.items():
        print(f"  [{gname:12s}]  score={r['score']:.4f}")
        print(f"    {r['readable']}")


if __name__ == "__main__":
    main()
