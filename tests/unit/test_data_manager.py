"""
tests/unit/test_data_manager.py — MT5DataManager 單元測試

驗證需求：
  - Req 3.5: 少於 MIN_BARS 的品種應被排除並記錄 WARNING

注意：測試使用較小的數據量（100/2000 bars），需同時 patch Config.MIN_BARS=100
以避免受全局配置（3000）影響。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# 測試用的 MIN_BARS 值（與測試數據大小匹配）
_TEST_MIN_BARS = 100


# ── 輔助函數 ──────────────────────────────────────────────────────────────────

def _make_ohlcv_df(n_rows: int, start_time: int = 1_000_000) -> pd.DataFrame:
    """構造含 n_rows 行的標準 OHLCV DataFrame。

    列：time, open, high, low, close, tick_volume
    time 為唯一遞增整數（Unix 時間戳風格）。
    """
    times = np.arange(start_time, start_time + n_rows, dtype=np.int64)
    opens = np.random.uniform(100.0, 200.0, size=n_rows)
    highs = opens + np.random.uniform(0.1, 5.0, size=n_rows)
    lows  = opens - np.random.uniform(0.1, 5.0, size=n_rows)
    closes = opens + np.random.uniform(-2.0, 2.0, size=n_rows)
    volumes = np.random.randint(100, 10_000, size=n_rows).astype(np.int64)

    return pd.DataFrame({
        "time":        times,
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "tick_volume": volumes,
    })


def _make_mock_fetcher(return_map: dict) -> MagicMock:
    """構造 MT5DataFetcher mock，根據 symbol 返回不同 DataFrame。

    Args:
        return_map: {symbol: pd.DataFrame}
    """
    fetcher = MagicMock()

    def _fetch_side_effect(symbol, timeframe, count):
        if symbol in return_map:
            return return_map[symbol]
        # 默認返回空 DataFrame
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "tick_volume"]
        )

    fetcher.fetch.side_effect = _fetch_side_effect
    return fetcher


# ── 測試 1：少於 100 bars 的品種被排除 ────────────────────────────────────────

class TestSymbolExcludedWhenBelowMinBars:
    """Req 3.5: 數據不足 MIN_BARS(100) 的品種必須被排除。"""

    def test_symbol_with_fewer_than_100_bars_is_excluded(self):
        """US500 只有 50 行，應被排除；XAUUSD 和 EURUSD 各有 2000 行，應保留。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(2000, start_time=1_000_000),
            "US500":  _make_ohlcv_df(50,   start_time=2_000_000),   # < 100
            "EURUSD": _make_ohlcv_df(2000, start_time=1_000_000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        assert "US500"  not in manager.symbols, "US500 應因 bars < 100 被排除"
        assert "XAUUSD" in manager.symbols,     "XAUUSD 有 2000 bars，應保留"
        assert "EURUSD" in manager.symbols,     "EURUSD 有 2000 bars，應保留"

    def test_excluded_symbol_fetch_was_called(self):
        """即使 US500 被排除，fetcher.fetch() 也應被調用過（先獲取後過濾）。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(2000),
            "US500":  _make_ohlcv_df(50),
            "EURUSD": _make_ohlcv_df(2000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        # fetch 應被調用 3 次（每個品種一次）
        assert mock_fetcher.fetch.call_count == 3

    def test_valid_symbols_count_after_exclusion(self):
        """排除 US500 後，manager.symbols 應只有 2 個有效品種。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(2000),
            "US500":  _make_ohlcv_df(10),   # 遠低於 100
            "EURUSD": _make_ohlcv_df(500),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        assert len(manager.symbols) == 2


# ── 測試 2：所有品種都不足 100 bars 時拋出 ValueError ─────────────────────────

class TestAllSymbolsBelowMinBarsRaisesError:
    """Req 3.5: 若所有品種均不滿足 MIN_BARS，應拋出 ValueError。"""

    def test_raises_value_error_when_all_symbols_below_min_bars(self):
        """所有品種返回 < 100 行數據時，load() 必須拋出 ValueError。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(50),
            "US500":  _make_ohlcv_df(30),
            "EURUSD": _make_ohlcv_df(1),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            with pytest.raises(ValueError) as exc_info:
                manager.load()

        # 錯誤消息應提示無可用品種
        assert "No valid symbols" in str(exc_info.value) or \
               "fewer than" in str(exc_info.value) or \
               "MIN_BARS" in str(exc_info.value)

    def test_raises_value_error_with_empty_dataframes(self):
        """所有品種返回空 DataFrame（0 行）時，load() 也應拋出 ValueError。"""
        empty_df = pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "tick_volume"]
        )
        fetch_map = {
            "XAUUSD": empty_df,
            "EURUSD": empty_df,
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "EURUSD"]):
            with pytest.raises(ValueError):
                manager.load()


# ── 測試 3：恰好 100 bars 的品種應被保留（邊界值）────────────────────────────

class TestExactlyMinBarsIsAccepted:
    """Req 3.5: MIN_BARS = 100，恰好 100 bars 的品種不應被排除。"""

    def test_exactly_100_bars_is_included(self):
        """品種返回恰好 100 行（= MIN_BARS）時，應被包含在 manager.symbols 中。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(100),  # 恰好等於 MIN_BARS
            "EURUSD": _make_ohlcv_df(2000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "EURUSD"]):
            manager.load()

        assert "XAUUSD" in manager.symbols, \
            "恰好 100 bars（= MIN_BARS）的品種應被保留，不應被排除"
        assert "EURUSD" in manager.symbols

    def test_99_bars_is_excluded_but_100_is_included(self):
        """99 bars（< MIN_BARS）應被排除，100 bars（= MIN_BARS）應保留——邊界嚴格區分。"""
        fetch_map = {
            "XAUUSD": _make_ohlcv_df(99),   # 比 MIN_BARS 少 1
            "US500":  _make_ohlcv_df(100),  # 恰好等於 MIN_BARS
            "EURUSD": _make_ohlcv_df(2000),
        }
        mock_fetcher = _make_mock_fetcher(fetch_map)

        from data_pipeline.data_manager import MT5DataManager
        from config import Config

        manager = MT5DataManager(mock_fetcher)

        with patch.object(Config, "MIN_BARS", _TEST_MIN_BARS), patch.object(Config, "SYMBOLS", ["XAUUSD", "US500", "EURUSD"]):
            manager.load()

        assert "XAUUSD" not in manager.symbols, "99 bars 應被排除"
        assert "US500"  in manager.symbols,     "100 bars 應被保留"
        assert "EURUSD" in manager.symbols,     "2000 bars 應被保留"
