"""
data_pipeline/fetcher.py — MT5 數據獲取模組

通過 MetaTrader5 Python API 連接 MT5 終端並獲取歷史 OHLCV 數據。
"""

import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

# DataFrame 返回列定義
_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


class MT5DataFetcher:
    """通過 MetaTrader5 Python API 獲取歷史 OHLCV 數據。

    用法（上下文管理器）：
        with MT5DataFetcher() as fetcher:
            df = fetcher.fetch("XAUUSD", mt5.TIMEFRAME_H1, 2000)

    用法（手動）：
        fetcher = MT5DataFetcher()
        fetcher.connect()
        df = fetcher.fetch("XAUUSD", mt5.TIMEFRAME_H1, 2000)
        fetcher.shutdown()

    離線模式（僅讀本地快取，不連 MT5）：
        with MT5DataFetcher(offline=True) as fetcher:
            df = fetcher.fetch("XAUUSD", mt5.TIMEFRAME_H1, 2000)
    """

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline
        self._mt5_initialized = False

    def connect(self) -> None:
        """連接到 MT5 終端。

        調用 `mt5.initialize()`，若連接失敗則拋出 `ConnectionError`。
        離線模式下跳過連接，僅使用本地快取。

        Raises:
            ConnectionError: MT5 終端未運行或連接失敗（非離線模式）。
        """
        if self.offline:
            logger.info("[Fetcher] 離線模式：跳過 MT5 連接，僅使用本地快取。")
            return

        if not _MT5_AVAILABLE:
            raise ConnectionError("MetaTrader5 package is not installed.")

        success = mt5.initialize()  # type: ignore[union-attr]
        if not success:
            error = mt5.last_error()  # type: ignore[union-attr]
            raise ConnectionError(f"MT5 connection failed: {error}")

        self._mt5_initialized = True
        logger.info("MT5 connection established.")

    def fetch(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """獲取指定品種的歷史 OHLCV 數據（優先讀本地快取，增量更新）。

        Args:
            symbol:    MT5 品種標識符，例如 "XAUUSD"。
            timeframe: MT5 時間週期常量，例如 mt5.TIMEFRAME_H1（整數）。
            count:     要獲取的 K 線數量。

        Returns:
            包含列 time, open, high, low, close, tick_volume 的 DataFrame。
            若品種不可用，返回空 DataFrame（列名相同）。
        """
        # ── 優先讀本地快取 ────────────────────────────────────────────
        # 使用本地快取的全部歷史數據，不再用 tail(count) 截斷。
        # count 僅用於無本地快取時從 MT5 全量下載的最大根數。
        try:
            from data_pipeline.kline_cache import KlineCache
            cache = KlineCache(timeframe=timeframe, bars_count=count)
            mt5_connected = (
                not self.offline
                and self._mt5_initialized
                and _MT5_AVAILABLE
                and mt5 is not None
            )
            df = cache.get(symbol, mt5_connected=mt5_connected)
            if df is not None and not df.empty:
                # 本地有數據，返回全部歷史（不截斷）
                return df.reset_index(drop=True)
        except Exception as exc:
            logger.debug(f"[Fetcher] Cache read failed for {symbol}: {exc}, falling back to MT5")

        # ── 快取不足時從 MT5 直接拉 ──────────────────────────────────
        if self.offline or not _MT5_AVAILABLE or mt5 is None or not self._mt5_initialized:
            logger.warning(f"{'Offline mode' if self.offline else 'MT5 not available'}, "
                           f"returning empty DataFrame for {symbol}.")
            return pd.DataFrame(columns=_COLUMNS)

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)  # type: ignore[union-attr]

        if rates is None or len(rates) == 0:
            logger.warning(
                f"Symbol '{symbol}' returned no data (possibly unavailable). "
                f"MT5 error: {mt5.last_error()}"  # type: ignore[union-attr]
            )
            return pd.DataFrame(columns=_COLUMNS)

        df = pd.DataFrame(rates)[_COLUMNS]
        logger.debug(f"Fetched {len(df)} bars for {symbol} (timeframe={timeframe}) from MT5.")
        return df

    def shutdown(self) -> None:
        """斷開與 MT5 終端的連接，釋放資源。"""
        if not self.offline and self._mt5_initialized and _MT5_AVAILABLE and mt5 is not None:
            mt5.shutdown()  # type: ignore[union-attr]
            logger.info("MT5 connection closed.")

    # ── 上下文管理器支持 ──────────────────────────────────

    def __enter__(self) -> "MT5DataFetcher":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
