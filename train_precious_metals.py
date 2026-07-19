"""
train_precious_metals.py — 貴金屬策略訓練（XAUUSD + XAGUSD）
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
    print(f"  AlphaGPT 訓練 — precious_metals 組 (貴金屬)")
    print(f"{'='*60}")
    print(f"  品種: {Config.SYMBOL_GROUPS['precious_metals']}")
    print(f"  獎勵模式: {ModelConfig.REWARD_MODE}")
    print(f"  訓練步數: {ModelConfig.TRAIN_STEPS}")
    print(f"  offline={offline}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        gsyms = Config.SYMBOL_GROUPS["precious_metals"]
        eng = train_group(fetcher, "precious_metals", gsyms, offline)
        if eng is not None:
            print(f"\n<<< [precious_metals] 完成: score={eng.best_score:.4f}")
            print(f"    {eng._decode_formula(eng.best_formula)}")
        else:
            print("\n<<< [precious_metals] 失敗")

    elapsed = time.time() - t0
    print(f"\n耗時 {elapsed/3600:.2f}h")

if __name__ == "__main__":
    main()
