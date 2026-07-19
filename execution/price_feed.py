"""
execution/price_feed.py — MT5 即時價格獲取模組

MT5PriceFeed 負責通過 MetaTrader5 Python API 獲取指定品種的最新 bid/ask 報價。
全同步介面，無 asyncio（Req 7.4）。
"""
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    # 測試環境占位，無需真實 MT5 安裝
    class _MT5Stub:
        def symbol_info_tick(self, symbol):  # noqa: D401
            return None

        def last_error(self):
            return (0, "MT5 not available")

    mt5 = _MT5Stub()

from loguru import logger


class MT5PriceFeed:
    """從 MT5 終端獲取即時 bid/ask/mid 報價。

    所有方法均為同步調用，符合 MetaTrader5 Python API 同步特性（Req 7.4）。
    """

    @staticmethod
    def get_tick(symbol: str) -> dict | None:
        """獲取指定品種的最新報價。

        調用 ``mt5.symbol_info_tick(symbol)`` 取得當前 tick 數據，計算 mid
        價格並以字典形式返回（Req 7.1、7.2）。

        若 tick 數據無法獲取（symbol 不存在、MT5 未連接等），記錄警告日誌並
        返回 None（Req 7.3）。無論日誌記錄本身是否成功，均保證返回 None。

        Args:
            symbol: MT5 品種標識符，例如 ``"XAUUSD"``、``"EURUSD"``。

        Returns:
            成功時返回::

                {"bid": float, "ask": float, "mid": float}

            失敗時返回 ``None``。
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            try:
                logger.warning(
                    f"MT5PriceFeed: symbol_info_tick('{symbol}') returned None"
                )
            except Exception:  # pragma: no cover — 日誌失敗不影響返回值
                pass
            return None

        bid: float = float(tick.bid)
        ask: float = float(tick.ask)
        mid: float = (bid + ask) / 2.0

        return {"bid": bid, "ask": ask, "mid": mid}
