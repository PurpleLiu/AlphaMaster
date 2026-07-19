"""
tests/unit/test_fetcher.py — MT5DataFetcher 邊界條件單元測試

驗證需求：
  - Req 2.2: mt5.initialize() 返回 False 時拋出 ConnectionError
  - Req 2.7: symbol 不可用（copy_rates_from_pos 返回 None 或空列表）時記錄 WARNING 並返回空 DataFrame
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# ── 輔助函數 ──────────────────────────────────────────────────────────────────

def _make_mt5_mock(**kwargs) -> MagicMock:
    """創建一個預配置的 mt5 MagicMock，接受任意關鍵字覆蓋預設值。"""
    mock = MagicMock()
    # 預設值：成功連接、copy_rates_from_pos 返回 None
    mock.initialize.return_value = kwargs.get("initialize", True)
    mock.last_error.return_value = kwargs.get("last_error", (0, "OK"))
    mock.copy_rates_from_pos.return_value = kwargs.get("copy_rates_from_pos", None)
    mock.shutdown.return_value = None
    return mock


# ── 測試 1：mt5.initialize() 返回 False → 拋出 ConnectionError ────────────────

class TestConnectRaisesOnFailure:
    """Req 2.2: 連接失敗時應拋出 ConnectionError。"""

    def test_connection_error_when_initialize_false(self):
        """當 mt5.initialize() 返回 False 時，connect() 必須拋出 ConnectionError。"""
        mt5_mock = _make_mt5_mock(initialize=False, last_error=(1, "Test error"))

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()

            with pytest.raises(ConnectionError) as exc_info:
                fetcher.connect()

        # 錯誤消息應包含 MT5 錯誤詳情
        assert "MT5 connection failed" in str(exc_info.value)

    def test_connection_error_message_contains_last_error(self):
        """ConnectionError 消息應包含 mt5.last_error() 返回的錯誤資訊。"""
        error_tuple = (5, "Terminal not found")
        mt5_mock = _make_mt5_mock(initialize=False, last_error=error_tuple)

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()

            with pytest.raises(ConnectionError) as exc_info:
                fetcher.connect()

        assert "Terminal not found" in str(exc_info.value) or \
               str(error_tuple) in str(exc_info.value)


# ── 測試 2：copy_rates_from_pos 返回 None → 返回空 DataFrame ─────────────────

class TestFetchReturnsEmptyDataFrameOnNone:
    """Req 2.7: symbol 不可用時返回含正確列的空 DataFrame。"""

    EXPECTED_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]

    def test_returns_empty_dataframe_when_rates_is_none(self):
        """copy_rates_from_pos 返回 None 時，fetch() 應返回空 DataFrame。"""
        mt5_mock = _make_mt5_mock(
            initialize=True,
            copy_rates_from_pos=None,
            last_error=(0, "OK"),
        )

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()
            df = fetcher.fetch("NOSUCHSYMBOL", 1, 100)

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == self.EXPECTED_COLUMNS

    def test_empty_dataframe_has_correct_columns_when_rates_is_none(self):
        """返回的空 DataFrame 列順序必須與 _COLUMNS 定義一致。"""
        mt5_mock = _make_mt5_mock(copy_rates_from_pos=None)

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()
            df = fetcher.fetch("XAUUSD", 16385, 500)

        assert set(df.columns) == set(self.EXPECTED_COLUMNS)
        assert len(df) == 0


# ── 測試 3：copy_rates_from_pos 返回空列表/數組 → 返回空 DataFrame ────────────

class TestFetchReturnsEmptyDataFrameOnEmptyRates:
    """Req 2.7: copy_rates_from_pos 返回空列表時同樣返回空 DataFrame。"""

    EXPECTED_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]

    def test_returns_empty_dataframe_when_rates_is_empty_list(self):
        """copy_rates_from_pos 返回空列表 [] 時，fetch() 應返回空 DataFrame。"""
        mt5_mock = _make_mt5_mock(copy_rates_from_pos=[])

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()
            df = fetcher.fetch("NOSUCHSYMBOL", 1, 100)

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == self.EXPECTED_COLUMNS

    def test_returns_empty_dataframe_when_rates_is_empty_array(self):
        """copy_rates_from_pos 返回空 numpy 數組時，fetch() 應返回空 DataFrame。"""
        import numpy as np
        empty_array = np.array([])
        mt5_mock = _make_mt5_mock(copy_rates_from_pos=empty_array)

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()
            df = fetcher.fetch("EURUSD", 16385, 200)

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == self.EXPECTED_COLUMNS

    def test_empty_dataframe_columns_match_spec(self):
        """列名必須完全匹配規範定義的六列，不多不少。"""
        mt5_mock = _make_mt5_mock(copy_rates_from_pos=[])

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()
            df = fetcher.fetch("US500", 16408, 1000)

        assert len(df.columns) == 6
        for col in ["time", "open", "high", "low", "close", "tick_volume"]:
            assert col in df.columns, f"缺少列: {col}"
