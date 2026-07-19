"""
train_index.py — 指數策略訓練（US30, US100, US500, US2000, JP225）
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from main import train_group

def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 訓練 — index 組 (美國+日本指數)")
    print(f"{'='*60}")
    print(f"  品種: {Config.SYMBOL_GROUPS['index']}")
    print(f"  獎勵模式: {ModelConfig.REWARD_MODE}")
    print(f"  訓練步數: {ModelConfig.TRAIN_STEPS}")
    print(f"  offline={offline}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        gsyms = Config.SYMBOL_GROUPS["index"]
        eng = train_group(fetcher, "index", gsyms, offline)
        if eng is not None:
            print(f"\n<<< [index] 完成: score={eng.best_score:.4f}")
            print(f"    {eng._decode_formula(eng.best_formula)}")
        else:
            print("\n<<< [index] 失敗")

    elapsed = time.time() - t0
    print(f"\n耗時 {elapsed/3600:.2f}h")

if __name__ == "__main__":
    main()
