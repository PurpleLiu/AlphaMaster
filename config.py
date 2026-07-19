"""
config.py — 統一配置模組（項目根目錄）

所有子模組從此文件導入 Config，廢棄各自的 config.py。
MT5 連接憑證通過環境變數或 .env 文件載入。
"""
import os

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    # 測試環境無 MT5 時使用整數占位常量（與真實 MT5 值一致）
    class _MT5Stub:
        TIMEFRAME_M1  = 1
        TIMEFRAME_M5  = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1  = 16385
        TIMEFRAME_H4  = 16388
        TIMEFRAME_D1  = 16408
        TIMEFRAME_W1  = 32769
        TIMEFRAME_MN1 = 49153
    mt5 = _MT5Stub()

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── MT5 連接 ──────────────────────────────────────────
    MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER   = os.getenv("MT5_SERVER", "")

    # ── 品種與週期 ────────────────────────────────────────
    # TRADE_SYMBOLS：實際交易的品種（新帳號，無 m 後綴）
    SYMBOLS   = [
        # 外匯
        "EURUSD", "USDJPY",
        # 貴金屬
        "XAUUSD", "XAGUSD",
        # AAVUSD 已移除：實為 Aave 加密貨幣，非大宗商品。波動特徵與貴金屬完全不匹配。
        # COCOA.c 已移除：大宗商品期貨，日交易~10h，時間對齊後僅8546 bar，
        # 拖累整組數據量，且流動性/交易時段與貴金屬不匹配。
        # 美國指數
        "US30.cash", "US100.cash", "US500.cash", "US2000.cash",
        # 其他指數
        "JP225.cash",
    ]

    # ── 訓練品種（單品種模式）──────────────────────────────
    # 2026-07-07 從分組模式切換到單品種模式。原因：
    #   1. 截面資訊沒用上：precious_metals/index 跑出的4個最優公式，沒有一個用了 CS 運算元
    #   2. 跨品種干擾嚴重：XAUUSD/XAGUSD 同組時，白銀 Kyle Lambda 波動是黃金4倍，
    #      模型被白銀主導，黃金信號被淹沒
    #   3. 指數組無效探索：5個美股指數相關性>0.85，截面空間狹窄，19次重啟後仍是beta
    #   4. 單品種策略更純粹：因子只針對一種資產特徵，實盤也更容易管理
    # 每個品種獨立訓練，checkpoint 按 ckpt_{symbol}_step_{N}.pt 保存。
    TRAINABLE_SYMBOLS = [
        # 外匯（各8年數據，24h連續交易）
        "EURUSD",
        "USDJPY",
        # 貴金屬（8年數據，24h連續交易）
        "XAUUSD",
        # XAGUSD 已移除：白銀與黃金高度相關，單品種訓練收益有限，
        # 且黃金已有驗證策略(Sharpe 2.66)，優先覆蓋未挖掘品種
        # 美國指數（各5年數據）
        "US30.cash",
        "US100.cash",
        "US500.cash",
        "US2000.cash",
        # 日本指數
        "JP225.cash",
    ]

    # [deprecated] 相關性分組（已廢棄，改用 TRAINABLE_SYMBOLS 單品種訓練）
    # 保留供回測參考，新訓練不再使用
    SYMBOL_GROUPS = {
        "forex":          ["EURUSD", "USDJPY"],
        "precious_metals":["XAUUSD", "XAGUSD"],
        "index":          ["US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"],
    }

    # FEATURE_SYMBOLS：用於計算截面特徵的寬品種集
    # 包含主要外匯、貴金屬、大宗商品、主流指數，時間與 SYMBOLS 高度對齊
    # REL_RET5/REL_RET20/REL_VOL 等跨資產特徵將基於這 40 個品種計算截面均值
    # 若設為 None，則退化為只用 SYMBOLS（5品種截面）
    FEATURE_SYMBOLS = [
        # 主要外匯（26個）
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
        "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURAUD",
        "EURCAD", "EURCHF", "GBPAUD", "GBPCAD", "GBPCHF",
        "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "NZDJPY",
        "CADCHF", "CADJPY", "CHFJPY",
        # 貴金屬（3個）
        "XAUUSD", "XAGUSD", "XPTUSD",
        # 美元指數（1個）
        "DXY.cash",
        # 大宗商品（2個）
        "USOIL.cash", "UKOIL.cash",
        # 主流指數（8個）
        "US30.cash", "US500.cash", "US100.cash", "UK100.cash",
        "DE30.cash", "FR40.cash", "JP225.cash", "AUS200.cash",
    ]

    # ── 數據參數 ──────────────────────────────────────────
    TIMEFRAME             = mt5.TIMEFRAME_H1   # K 線週期
    # 每品種拉取的歷史 K 線上限。設為極大值以使用 MT5 全部可用歷史；
    # 本地快取優先：若 D:\K線數據 已有數據，fetcher 會返回本地全部歷史（不截斷）。
    BARS_COUNT            = 10_000_000
    MIN_BARS              = 3000   # 低於此值的品種被排除
    DATA_REFRESH_INTERVAL = 300    # 秒，實盤數據刷新間隔
    KLINE_CACHE_DIR       = os.getenv("KLINE_CACHE_DIR", r"D:\K線數據")  # 本地 K 線快取目錄

    # ── 模型參數（僅供參考，訓練實際使用 model_core.config.ModelConfig）────
    # 訓練參數的權威來源是 model_core/config.py，這裡的值不生效
    INPUT_DIM       = 20           # 特徵數（與 MT5FeatureEngineer.INPUT_DIM 一致）
    BATCH_SIZE      = 128          # 參見 ModelConfig.BATCH_SIZE
    TRAIN_STEPS     = 300          # 參見 ModelConfig.TRAIN_STEPS
    MAX_FORMULA_LEN = 8            # 參見 ModelConfig.MAX_FORMULA_LEN
    # DEVICE 同樣以 model_core/config.py 為準（已改為 cpu，原因見該文件注釋）
    DEVICE          = (
        torch.device("cpu")
        if _TORCH_AVAILABLE
        else "cpu"
    )

    # ── 風控參數 ──────────────────────────────────────────
    RISK_PER_TRADE     = 0.01      # legacy: 保留給舊介面/測試；實盤倉位使用 VOL_TARGET_* 參數
    COST_RATE          = 0.0001    # 單邊點差+佣金（forex/metals）
    MAX_OPEN_POSITIONS = 4         # 最多同時持倉品種數
    MAX_LOT_PER_TRADE  = 5.0       # 兜底上限；實際手數由 XAUUSD 0.01 手波動預算決定
    # 永不自動交易的品種（白銀合約乘數 5000，2026-07-08 起停用）
    EXCLUDED_TRADE_SYMBOLS = ["XAGUSD"]
    # 手數校準（實盤）：
    # - 以 XAUUSD 0.01 手的一根 ATR 美元波動作為基準
    # - 其它品種按各自 ATR 與 tick value 反推手數，使金額波動接近
    # - 可選 Sharpe 權重：Sharpe 高於基準則略放大，低於基準則收縮
    FIXED_LOT_BY_SYMBOL = {
        "XAUUSD": 0.01,
    }
    VOL_TARGET_REFERENCE_SYMBOL = "XAUUSD"
    VOL_TARGET_REFERENCE_LOT = 0.01
    VOL_TARGET_SHARPE_REFERENCE = 2.447
    VOL_TARGET_SHARPE_EXPONENT = 0.50
    VOL_TARGET_MIN_SHARPE_WEIGHT = 0.50
    VOL_TARGET_MAX_SHARPE_WEIGHT = 1.50
    VOL_TARGET_SHARPE_BY_SYMBOL = {
        "XAUUSD": 2.447,
        "US100.cash": 1.811,
        "US500.cash": 0.959,
        "US2000.cash": 0.575,
        "US30.cash": 0.923,
        "JP225.cash": -0.653,
    }
    MIN_TRADE_EXPOSURE = 0.05      # |tanh(factor)| 小於該值時視為空倉，回測/實盤共用

    # ── 策略參數 ──────────────────────────────────────────
    # SIGNAL_MODE 控制信號→倉位的轉換方式：
    #   "backtest_parity": tanh 連續倉位，與 backtest.py 完全一致（推薦）
    #   "threshold":       sigmoid + BUY_THRESHOLD / SELL_THRESHOLD（舊邏輯）
    SIGNAL_MODE = "backtest_parity"

    # EXIT_MODE 控制出場機制：
    #   "signal":  僅靠信號翻轉出場，嚴格對標回測
    #   "risk":    保留止損/止盈/追蹤止損
    #   "hybrid":  信號翻轉為主，保留緊急熔斷（單日最大虧損 / 極端滑點）
    EXIT_MODE = "signal"

    # threshold 模式專用（SIGNAL_MODE="threshold" 時生效）
    BUY_THRESHOLD       = 0.70
    SELL_THRESHOLD      = 0.40

    # risk / hybrid 模式專用（EXIT_MODE != "signal" 時生效）
    STOP_LOSS_PCT       = -0.02   # -2%
    TAKE_PROFIT_PCT     = 0.04    # +4%
    TRAILING_ACTIVATION = 0.03
    TRAILING_DROP       = 0.015

    # 時間對齊
    REBALANCE_ON_BAR_CLOSE = True  # True=僅新 K 線收盤後調倉，對標回測
    EXECUTION_LAG_BARS     = 1     # 與回測 target_ret 的執行延遲對齊

    # 持倉上限：None = 不限制（嚴格對標回測，各品種獨立）
    # 設為整數（如 3）則啟用約束（需回測裡同步加同樣約束才對標）
    MAX_OPEN_POSITIONS: int | None = None

    # ── 文件路徑 ──────────────────────────────────────────
    STRATEGY_FILE  = "best_mt5_strategy.json"
    PORTFOLIO_FILE = "portfolio_state.json"
    STOP_SIGNAL    = "STOP_SIGNAL"

    # ── Magic Number ──────────────────────────────────────
    MAGIC_NUMBER = 20250101

    @classmethod
    def get_timeframe(cls, tf_str: str) -> int:
        """將字串（如 'H1'）映射為 MT5 時間週期常量。

        支持的週期：M1, M5, M15, M30, H1, H4, D1, W1, MN1

        Args:
            tf_str: 時間週期字串，例如 "H1"

        Returns:
            對應的 MT5 TIMEFRAME_* 整數常量

        Raises:
            ValueError: 若 tf_str 不在支持列表中
        """
        mapping = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        if tf_str not in mapping:
            raise ValueError(
                f"Unknown timeframe: '{tf_str}'. "
                f"Supported values: {list(mapping.keys())}"
            )
        return mapping[tf_str]
