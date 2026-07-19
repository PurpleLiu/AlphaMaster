"""
tests/unit/test_portfolio.py — MT5PortfolioManager 單元測試

涵蓋：
  - add_position() 記錄 MT5 ticket（Req 8.2）
  - sync_from_mt5() 移除 MT5 中已不存在的倉位（Req 8.4, 8.5）
  - sync_from_mt5() 保留 MT5 中仍然存在的倉位（Req 8.4, 8.5）
  - sync_from_mt5() 在 positions_get() 返回 None 時不崩潰（Req 8.4）
"""
import os
from unittest.mock import MagicMock, patch

import strategy_manager.portfolio as portfolio_module
from strategy_manager.portfolio import MT5PortfolioManager


# ─────────────────────────────────────────────────────────────────
# 輔助：構造一個指向臨時路徑的 portfolio manager
# ─────────────────────────────────────────────────────────────────

def _make_manager(tmp_path) -> MT5PortfolioManager:
    """返回使用臨時狀態文件的 MT5PortfolioManager。

    通過在 __init__ 完成後立即重設 state_file 並清空 positions，
    確保測試互相隔離，不汙染磁碟。
    """
    # load_state() 在 __init__ 中被調用，此時會嘗試讀取 Config.PORTFOLIO_FILE
    # 若文件不存在，它會靜默地初始化空 positions —— 沒有副作用。
    manager = MT5PortfolioManager()
    manager.state_file = str(tmp_path / "portfolio_state.json")
    manager.positions = {}
    return manager


# ─────────────────────────────────────────────────────────────────
# 測試 1：add_position() 記錄 ticket（Req 8.2）
# ─────────────────────────────────────────────────────────────────

def test_add_position_records_ticket(tmp_path):
    """add_position() 必須將 ticket 號碼正確記錄到 Position.ticket。"""
    manager = _make_manager(tmp_path)

    manager.add_position("XAUUSD", ticket=123456, price=1900.0, lot=0.01, direction="BUY")

    assert "XAUUSD" in manager.positions, "倉位應被記錄到 positions 字典中"
    assert manager.positions["XAUUSD"].ticket == 123456, (
        f"期望 ticket=123456，實際為 {manager.positions['XAUUSD'].ticket}"
    )


# ─────────────────────────────────────────────────────────────────
# 測試 2：sync_from_mt5() 移除 MT5 中不存在的 ticket（Req 8.4, 8.5）
# ─────────────────────────────────────────────────────────────────

def test_sync_removes_position_when_ticket_missing_from_mt5(tmp_path):
    """mt5.positions_get() 不返回 ticket=123456 時，XAUUSD 倉位應被移除。"""
    manager = _make_manager(tmp_path)
    manager.add_position("XAUUSD", ticket=123456, price=1900.0, lot=0.01, direction="BUY")
    assert "XAUUSD" in manager.positions

    # mt5.positions_get() 返回一個不含 ticket=123456 的列表
    other_pos = MagicMock()
    other_pos.ticket = 999999

    mock_mt5 = MagicMock()
    mock_mt5.positions_get.return_value = [other_pos]

    with patch.object(portfolio_module, "mt5", mock_mt5), \
         patch.object(portfolio_module, "_MT5_AVAILABLE", True):
        manager.sync_from_mt5()

    assert "XAUUSD" not in manager.positions, (
        "mt5.positions_get() 不含該 ticket 時，倉位應被移除"
    )


# ─────────────────────────────────────────────────────────────────
# 測試 3：sync_from_mt5() 保留 MT5 中仍存在的 ticket（Req 8.4, 8.5）
# ─────────────────────────────────────────────────────────────────

def test_sync_keeps_position_when_ticket_present_in_mt5(tmp_path):
    """mt5.positions_get() 含 ticket=789012 時，XAUUSD 倉位應被保留。"""
    manager = _make_manager(tmp_path)
    manager.add_position("XAUUSD", ticket=789012, price=2000.0, lot=0.02, direction="BUY")
    assert "XAUUSD" in manager.positions

    live_pos = MagicMock()
    live_pos.ticket = 789012

    mock_mt5 = MagicMock()
    mock_mt5.positions_get.return_value = [live_pos]

    with patch.object(portfolio_module, "mt5", mock_mt5), \
         patch.object(portfolio_module, "_MT5_AVAILABLE", True):
        manager.sync_from_mt5()

    assert "XAUUSD" in manager.positions, (
        "mt5.positions_get() 含該 ticket 時，倉位不應被移除"
    )


# ─────────────────────────────────────────────────────────────────
# 測試 4：sync_from_mt5() 在 positions_get() 返回 None 時不崩潰（Req 8.4）
# ─────────────────────────────────────────────────────────────────

def test_sync_handles_positions_get_returning_none(tmp_path):
    """mt5.positions_get() 返回 None 時，sync_from_mt5() 應靜默跳過，不崩潰，倉位不變。"""
    manager = _make_manager(tmp_path)
    manager.add_position("XAUUSD", ticket=111111, price=1950.0, lot=0.01, direction="SELL")

    mock_mt5 = MagicMock()
    mock_mt5.positions_get.return_value = None
    mock_mt5.last_error.return_value = (0, "no error")

    with patch.object(portfolio_module, "mt5", mock_mt5), \
         patch.object(portfolio_module, "_MT5_AVAILABLE", True):
        manager.sync_from_mt5()

    assert "XAUUSD" in manager.positions, (
        "positions_get() 返回 None 時，倉位不應被移除"
    )
