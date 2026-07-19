"""
tests/unit/test_runner.py — MT5StrategyRunner 單元測試

涵蓋：
  - 策略文件不存在時 → sys.exit(1) 被調用（Req 10.1）
  - shutdown() 調用 mt5.shutdown()（Req 10.7）
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

import strategy_manager.runner as runner_module
from strategy_manager.runner import MT5StrategyRunner


# ─────────────────────────────────────────────────────────────────
# 測試 1：策略文件不存在 → sys.exit(1)（Req 10.1）
# ─────────────────────────────────────────────────────────────────

def test_missing_strategy_file_causes_sys_exit():
    """當 best_mt5_strategy.json 不存在時，__init__() 必須調用 sys.exit(1)。

    通過 patch os.path.exists 返回 False 來模擬文件不存在的情況。
    pytest.raises(SystemExit) 捕獲 sys.exit() 拋出的 SystemExit 異常，
    並驗證退出碼為 1。
    """
    with patch("strategy_manager.runner.os.path.exists", return_value=False):
        with pytest.raises(SystemExit) as exc_info:
            MT5StrategyRunner()

    assert exc_info.value.code == 1, (
        f"期望退出碼為 1，實際為 {exc_info.value.code}"
    )


# ─────────────────────────────────────────────────────────────────
# 測試 2：shutdown() 調用 mt5.shutdown()（Req 10.7）
# ─────────────────────────────────────────────────────────────────

def test_shutdown_calls_mt5_shutdown():
    """shutdown() 必須調用 mt5.shutdown() 以釋放 MT5 連接。

    使用 __new__ 繞過 __init__（避免文件載入和子模組初始化），
    直接設置必要屬性後調用 shutdown()，驗證 mt5.shutdown() 被調用。
    """
    # 用 __new__ 創建實例，跳過 __init__
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)

    # 設置 shutdown() 所依賴的屬性
    runner.formula = [1, 2, 3]
    runner._fetcher = None  # shutdown() 中會檢查 _fetcher

    mock_mt5 = MagicMock()

    with patch.object(runner_module, "mt5", mock_mt5):
        runner.shutdown()

    mock_mt5.shutdown.assert_called_once(), "shutdown() 必須恰好調用一次 mt5.shutdown()"
