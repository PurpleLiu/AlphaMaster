"""
main.py — 多因子訓練入口（分組訓練）

使用方法：
    python main.py                      # 按 SYMBOL_GROUPS 分組訓練（默認推薦）
    python main.py --offline            # 僅使用本地快取，不連接 MT5
    python main.py --single XAUUSD      # 只訓練單個品種
    python main.py --cross-section      # 所有品種一起訓練（截面）
    python main.py --group risk         # 只訓練 risk 組

分組說明（Config.SYMBOL_GROUPS）：
    forex 組（EURUSD, USDJPY）：外匯，美元方向因子，2品種×11000=22000樣本
    risk  組（XAUUSD, US100.cash, US500.cash）：風險資產，3品種×11000=33000樣本
"""
import sys, pathlib, json

# 無控制台環境（如 Start-Process -WindowStyle Hidden）下，sys.stdout 可能為 None，
# 導致 tqdm.write 報錯。此時重定向到日誌檔案。
if sys.stdout is None or sys.stderr is None:
    _log_path = r"D:\素材\自動挖因子\training_stdout.log"
    _log_fp = open(_log_path, "a", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _log_fp
    if sys.stderr is None:
        sys.stderr = _log_fp

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.single_symbol_manager import SingleSymbolDataManager
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig
from model_core.vocab import VOCAB_VERSION


class GroupDataManager:
    """品種分組數據視圖，相容 AlphaEngine 介面（N = 組內品種數）。"""
    def __init__(self, multi_manager, symbols: list[str]):
        self._multi   = multi_manager
        self._symbols = [s for s in symbols if s in multi_manager.symbols]
        self._idxs    = [multi_manager.symbols.index(s) for s in self._symbols]

    @property
    def symbols(self): return list(self._symbols)

    @property
    def raw_dict(self):
        full = self._multi.raw_dict
        return {k: v[self._idxs] for k, v in full.items()}

    @property
    def feat_tensor(self):
        from model_core.features import MT5FeatureEngineer
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self):
        return self._multi.target_ret[self._idxs]

    @property
    def bar_time(self):
        return self._multi.bar_time[self._idxs]


def save_group_strategy(engine: AlphaEngine, group_name: str, symbols: list[str]):
    """保存分組策略：group 總文件 + 各品種 best_*.json。"""
    pathlib.Path("strategies").mkdir(exist_ok=True)

    # 分組總文件
    gp = pathlib.Path("strategies") / f"best_group_{group_name}.json"
    gp.write_text(json.dumps({
        "vocab_version": VOCAB_VERSION,
        "group":         group_name,
        "symbols":       symbols,
        "formula":       engine.best_formula,
        "best_score":    engine.best_score,
    }, indent=2))

    # 各品種文件（runner 按品種名載入）
    for sym in symbols:
        sp = pathlib.Path("strategies") / f"best_{sym}.json"
        sp.write_text(json.dumps({
            "vocab_version": VOCAB_VERSION,
            "symbol":        sym,
            "group":         group_name,
            "formula":       engine.best_formula,
            "best_score":    engine.best_score,
            "source":        f"group_{group_name}",
        }, indent=2))

    print(f"  已保存: best_group_{group_name}.json + {len(symbols)} 個品種文件")


def train_group(fetcher, group_name: str, symbols: list[str], offline: bool):
    """訓練一個品種組，使用組內獨立的 DataManager（不與其他組取時間交集）。

    自動檢測是否有可續訓的 checkpoint：若存在則從斷點繼續，否則從 step 0 開始。
    """
    print(f"\n{'─'*60}")
    print(f"  [{group_name}] 組: {symbols}")
    print(f"{'─'*60}")

    # 臨時覆蓋 SYMBOLS 只載入本組品種
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = [s for s in symbols if s in original_symbols or True]

    try:
        group_mgr = MT5DataManager(fetcher)
        group_mgr.load()
        actual_symbols = group_mgr.symbols
        T = group_mgr.raw_dict["open"].shape[1]
        print(f"  獨立載入: {actual_symbols}  T={T} bars")
    except Exception as e:
        print(f"  [錯誤] 數據載入失敗: {e}")
        Config.SYMBOLS = original_symbols
        return None
    finally:
        Config.SYMBOLS = original_symbols  # 恢復原始配置

    if not actual_symbols:
        print(f"  [跳過] 無有效品種")
        return None

    engine = AlphaEngine(data_manager=group_mgr, target_symbol=group_name)

    # ── 自動續訓：檢測最新 checkpoint ────────────────────────────────
    import glob as _glob
    ckpt_pattern = str(pathlib.Path("checkpoints") / f"ckpt_{group_name}_step_*.pt")
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    start_step = 0
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        try:
            start_step = engine.load_checkpoint(latest_ckpt)
            print(f"  [續訓] 從 {latest_ckpt} 恢復，start_step={start_step}")
        except Exception as e:
            print(f"  [警告] checkpoint 載入失敗（{e}），從頭開始訓練")
            start_step = 0
    else:
        print(f"  [新訓] 未找到 checkpoint，從 step 0 開始")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {group_name} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳過訓練")
        save_group_strategy(engine, group_name, actual_symbols)
        return engine

    engine.train(start_step=start_step)
    save_group_strategy(engine, group_name, actual_symbols)
    return engine


def main():
    cross    = "--cross-section" in sys.argv
    single   = "--single" in sys.argv
    offline  = "--offline" in sys.argv
    grp_only = None
    if "--group" in sys.argv:
        idx      = sys.argv.index("--group")
        grp_only = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    single_sym = [s for s in sys.argv[1:] if not s.startswith("--")] or None

    mode = "截面" if cross else ("單品種" if single else "分組")
    print(f"{'='*60}")
    print(f"  AlphaGPT 訓練 [{mode}模式]" + (" [離線快取]" if offline else ""))
    print(f"  TRAIN_STEPS={ModelConfig.TRAIN_STEPS}  "
          f"MAX_FORMULA_LEN={ModelConfig.MAX_FORMULA_LEN}  "
          f"BATCH_SIZE={ModelConfig.BATCH_SIZE}")
    print(f"  SYMBOLS={Config.SYMBOLS}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        if single and single_sym:
            # 單品種模式：用全局 multi_mgr
            multi_mgr = MT5DataManager(fetcher)
            multi_mgr.load()
            sym    = single_sym[0]
            s_mgr  = SingleSymbolDataManager(multi_mgr, sym)
            engine = AlphaEngine(data_manager=s_mgr, target_symbol=sym)
            engine.train()

        elif cross:
            multi_mgr = MT5DataManager(fetcher)
            multi_mgr.load()
            engine = AlphaEngine(data_manager=multi_mgr, target_symbol=None)
            engine.train()
            save_group_strategy(engine, "cross_section", multi_mgr.symbols)

        else:
            # 分組訓練：每組獨立載入，不共享 DataManager
            groups = getattr(Config, "SYMBOL_GROUPS", {
                "forex": ["EURUSD", "USDJPY"],
                "risk":  ["XAUUSD", "US100.cash", "US500.cash"],
            })
            if grp_only:
                groups = {grp_only: groups[grp_only]} if grp_only in groups else groups

            results = {}
            for gname, gsyms in groups.items():
                eng = train_group(fetcher, gname, gsyms, offline)
                if eng:
                    results[gname] = {
                        "score":   eng.best_score,
                        "formula": eng._decode_formula(eng.best_formula),
                    }

            print(f"\n{'='*60}")
            print(f"  分組訓練完成")
            print(f"{'='*60}")
            for gname, r in results.items():
                print(f"  [{gname}]: score={r['score']:.4f}")
                print(f"    {r['formula']}")
            print()


if __name__ == "__main__":
    main()
