"""
train_ftmo_island.py — FTMO 專屬 + Island 多起點並行訓練

訓練 index 組，使用 IslandAlphaEngine：
  - 3 個獨立模型（islands）
  - 每 200 步遷移一次 elite
  - 自適應噪聲、部分參數重設、elite decay 已啟用
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from model_core.island_engine import IslandAlphaEngine


def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  FTMO + Island 多起點並行訓練 — index 組")
    print(f"{'='*60}")
    print(f"  獎勵模式: {ModelConfig.REWARD_MODE}")
    print(f"  公式長度: {ModelConfig.MAX_FORMULA_LEN}")
    print(f"  Islands : {ModelConfig.N_ISLANDS}")
    print(f"  Migration: every {ModelConfig.MIGRATION_INTERVAL} steps, Top-{ModelConfig.MIGRATION_TOP_K}")
    print(f"  Adaptive noise: {ModelConfig.ADAPTIVE_NOISE}")
    print(f"  Partial reset: {ModelConfig.PARTIAL_RESET}")
    print(f"  Elite decay: {ModelConfig.ELITE_DECAY}")
    print(f"  品種: {Config.SYMBOL_GROUPS['index']}")
    print(f"{'='*60}\n")

    with MT5DataFetcher(offline=offline) as fetcher:
        # 臨時切換 SYMBOLS 到 index 組
        original_symbols = Config.SYMBOLS[:]
        Config.SYMBOLS = Config.SYMBOL_GROUPS["index"]
        try:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            engine = IslandAlphaEngine(mgr)
            engine.train()

            formula, score = engine.get_global_best()
            print(f"\n最終全局最優: score={score:.4f}")
            print(f"公式: {formula}")
        finally:
            Config.SYMBOLS = original_symbols

    elapsed = time.time() - t0
    print(f"\n總耗時 {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
