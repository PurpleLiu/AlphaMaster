"""
train_single.py — 單品種訓練入口

用法:
    python train_single.py XAUUSD [--offline]

每個品種獨立訓練，checkpoint: checkpoints/ckpt_{symbol}_step_{N}.pt
訓練完成後保存策略: strategies/best_{symbol}.json
"""
import sys, time, pathlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine


def train_single(fetcher, symbol: str, offline: bool):
    """訓練單個品種。

    自動檢測 checkpoint 續訓，完成後保存策略到 strategies/best_{symbol}.json。
    """
    import json, glob as _g

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 單品種訓練 — {symbol}")
    print(f"{'='*60}")
    print(f"  獎勵模式: {ModelConfig.REWARD_MODE}")
    print(f"  訓練步數: {ModelConfig.TRAIN_STEPS}")
    print(f"  K線目錄: {Config.KLINE_CACHE_DIR}")
    print(f"  離線模式: 是（僅讀本地快取）")
    print(f"{'='*60}")

    # 載入全部品種數據，然後切出單品種視圖
    original = Config.SYMBOLS[:]
    Config.SYMBOLS = [symbol]

    try:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  數據: {symbol}  T={T} bars ({T/6240:.2f}年)")
    except Exception as e:
        print(f"  [錯誤] 數據載入失敗: {e}")
        Config.SYMBOLS = original
        return None
    finally:
        Config.SYMBOLS = original

    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol)

    # 自動續訓
    ckpt_pattern = str(pathlib.Path("checkpoints") / f"ckpt_{symbol}_step_*.pt")
    ckpt_files = sorted(_g.glob(ckpt_pattern))
    start_step = 0

    if ckpt_files:
        latest = ckpt_files[-1]
        try:
            engine.load_checkpoint(latest)
            start_step = engine._step if hasattr(engine, '_step') else int(
                latest.split('_step_')[-1].replace('.pt', '')
            )
            print(f"  [續訓] 從 {latest} 恢復，start_step={start_step}")
        except Exception as e:
            print(f"  [警告] checkpoint 載入失敗: {e}，將從頭開始")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳過訓練")
        _save_strategy(engine, symbol)
        return engine

    if start_step == 0:
        print(f"  [新訓] 從 step 0 開始")

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol)
    return engine


def _save_strategy(engine, symbol):
    """保存單品種策略"""
    import json
    from model_core.vocab import VOCAB_VERSION

    pathlib.Path("strategies").mkdir(exist_ok=True)
    path = pathlib.Path("strategies") / f"best_{symbol}.json"

    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "mode": "single",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
            if engine.best_formula else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  策略已保存: {path}")


# ── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    offline = True
    mode = "ftmo"
    syms: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--mode":
            skip_next = True
            continue
        if arg == "--cache-dir":
            skip_next = True
            continue
        if arg.startswith("--"):
            continue
        syms.append(arg)

    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "ftmo"
    if "--cache-dir" in sys.argv:
        idx = sys.argv.index("--cache-dir")
        if idx + 1 < len(sys.argv):
            Config.KLINE_CACHE_DIR = sys.argv[idx + 1]
    ModelConfig.REWARD_MODE = mode

    if not syms:
        print("用法: python train_single.py <SYMBOL> [--cache-dir PATH] [--mode ftmo]")
        print(f"可選品種: {Config.TRAINABLE_SYMBOLS}")
        sys.exit(1)

    symbol = syms[0]
    if symbol not in Config.TRAINABLE_SYMBOLS:
        print(f"警告: {symbol} 不在 TRAINABLE_SYMBOLS 列表中，但仍將嘗試訓練")

    t0 = time.time()
    with MT5DataFetcher(offline=offline) as fetcher:
        eng = train_single(fetcher, symbol, offline)

    elapsed = time.time() - t0
    if eng:
        print(f"\n<<< [{symbol}] 完成: score={eng.best_score:.4f} 耗時 {elapsed/3600:.2f}h")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print(f"\n<<< [{symbol}] 失敗")
