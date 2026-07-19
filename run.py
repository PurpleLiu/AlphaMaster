"""
run.py — 實盤入口

啟動 MT5 實盤策略主循環。

使用方式：
    python run.py

前置條件：
    1. 已運行 main.py 完成訓練，生成 best_mt5_strategy.json
    2. 已配置 .env 文件，包含 MT5_LOGIN、MT5_PASSWORD、MT5_SERVER
    3. MetaTrader5 終端已啟動並登錄

流程：
    - 初始化 MT5StrategyRunner（載入策略公式，失敗時自動 sys.exit(1)）
    - 啟動同步主循環（runner.run()）
    - Ctrl+C（KeyboardInterrupt）可優雅中斷
    - try/finally 確保無論何種退出方式都調用 runner.shutdown()，釋放 MT5 連接

Requirements: 10.1–10.7
"""
from strategy_manager.runner import MT5StrategyRunner

if __name__ == "__main__":
    runner = MT5StrategyRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
