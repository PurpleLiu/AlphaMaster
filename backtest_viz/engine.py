"""
backtest_viz/engine.py — 逐 bar 可視化回測引擎

與訓練用 backtest.py 共享相同的信號邏輯（tanh 連續倉位），
但額外記錄每筆交易的開平倉細節，供圖表標註使用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from model_core.vm import StackVM
from strategy_manager.signal import target_to_direction

_H1_PERIODS_PER_YEAR = 6240


@dataclass
class Trade:
    """一筆完整交易記錄（開倉 → 平倉/反手）"""
    symbol:      str
    direction:   int          # +1 多 / -1 空
    entry_bar:   int          # 開倉 bar 索引（相對於整個序列）
    entry_time:  int          # Unix 秒
    entry_price: float        # 開倉價（用 open 價格）
    exit_bar:    Optional[int]   = None
    exit_time:   Optional[int]   = None
    exit_price:  Optional[float] = None
    pnl:         float           = 0.0   # 本筆稅後 PnL（log return - cost）
    cum_pnl:     float           = 0.0   # 截至本筆結束的累計 PnL


@dataclass
class SymbolResult:
    """單個品種的完整回測結果"""
    symbol:       str
    times:        np.ndarray     # Unix 秒，shape [T]
    open:         np.ndarray     # [T]
    high:         np.ndarray     # [T]
    low:          np.ndarray     # [T]
    close:        np.ndarray     # [T]
    volume:       np.ndarray     # [T]
    factor:       np.ndarray     # StackVM 輸出，[T]
    signal:       np.ndarray     # tanh(factor)，[T]
    position:     np.ndarray     # 連續倉位 ∈ [-1,+1]，[T]
    pnl:          np.ndarray     # 逐 bar PnL，[T]
    cum_pnl:      np.ndarray     # 累計 PnL，[T]
    trades:       list[Trade]    = field(default_factory=list)
    sortino:      float          = 0.0
    total_return: float          = 0.0
    n_trades:     int            = 0
    win_rate:     float          = 0.0
    max_drawdown: float          = 0.0
    avg_hold_bars:float          = 0.0
    profit_loss_ratio: float | None = None  # 盈虧比 = 平均盈利 / 平均虧損


class BacktestEngine:
    """逐 bar 可視化回測引擎。

    用法：
        engine = BacktestEngine(formula=[6,15,8,...])
        results = engine.run(raw_dict, times, symbols)
    """

    def __init__(
        self,
        formula:         list[int],
        cost_rate:       float = 0.0001,
        periods_per_year:int   = _H1_PERIODS_PER_YEAR,
    ):
        self.formula          = formula
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year
        self.vm               = StackVM()

    # ─────────────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        raw_dict: dict,          # {open/high/low/close/volume/time: Tensor[N,T]}
        feat_tensor: torch.Tensor,  # [N, F, T]
        symbols: list[str],
    ) -> list[SymbolResult]:
        """執行所有品種的回測，返回每個品種的 SymbolResult。"""

        factors_all = self.vm.execute(self.formula, feat_tensor)  # [N, T]
        if factors_all is None:
            raise RuntimeError(
                f"StackVM 無法執行公式 {self.formula}。"
                "請檢查公式 token 是否合法。"
            )

        results = []
        N = len(symbols)
        for n in range(N):
            sym = symbols[n]
            sym_result = self._backtest_symbol(
                symbol     = sym,
                raw_dict   = {k: v[n] for k, v in raw_dict.items()},   # [T] 各欄位
                factor_1d  = factors_all[n],                            # [T]
            )
            results.append(sym_result)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # 單品種回測
    # ─────────────────────────────────────────────────────────────────────

    def _backtest_symbol(
        self,
        symbol:   str,
        raw_dict: dict,         # 每個值是 [T] 的 Tensor
        factor_1d: torch.Tensor,  # [T]
    ) -> SymbolResult:

        T = factor_1d.shape[0]

        # numpy 轉換（便於後續圖表處理）
        factor_np   = factor_1d.detach().float().numpy()
        # 連續倉位模式：tanh 直接作為倉位比例，與訓練 backtest.py 完全一致
        signal_np   = np.tanh(factor_np)
        position_np = signal_np

        open_np   = raw_dict["open"].float().numpy()
        high_np   = raw_dict["high"].float().numpy()
        low_np    = raw_dict["low"].float().numpy()
        close_np  = raw_dict["close"].float().numpy()
        volume_np = raw_dict["volume"].float().numpy()

        if "time" in raw_dict:
            times_np = raw_dict["time"].long().numpy()
        else:
            times_np = np.arange(T, dtype=np.int64)

        # ── 計算 PnL 序列（與 backtest.py 完全一致）─────────────────
        # target_ret[t] = log(open[t+2] / open[t+1])
        target_ret = np.zeros(T, dtype=np.float32)
        if T >= 3:
            target_ret[: T - 2] = np.log(
                (open_np[2:] + 1e-12) / (open_np[1:-1] + 1e-12)
            )

        prev_pos = np.zeros(T, dtype=np.float32)
        prev_pos[1:] = position_np[:-1]
        turnover = np.abs(position_np - prev_pos)

        pnl_np    = position_np * target_ret - turnover * self.cost_rate
        cum_pnl   = np.cumsum(pnl_np)

        # ── 提取交易記錄 ──────────────────────────────────────────────
        trades = self._extract_trades(
            symbol, position_np, open_np, times_np, pnl_np
        )

        # ── 統計指標 ─────────────────────────────────────────────────
        sortino       = self._calc_sortino(pnl_np)
        total_return  = float(cum_pnl[-1]) if len(cum_pnl) else 0.0
        n_trades      = len(trades)
        win_rate      = (
            sum(1 for t in trades if t.pnl > 0) / n_trades
            if n_trades else 0.0
        )
        avg_hold      = (
            sum(
                (t.exit_bar - t.entry_bar)
                for t in trades if t.exit_bar is not None
            ) / n_trades
            if n_trades else 0.0
        )
        pl_ratio      = self._calc_profit_loss_ratio(trades)

        return SymbolResult(
            symbol       = symbol,
            times        = times_np,
            open         = open_np,
            high         = high_np,
            low          = low_np,
            close        = close_np,
            volume       = volume_np,
            factor       = factor_np,
            signal       = signal_np,
            position     = position_np,
            pnl          = pnl_np,
            cum_pnl      = cum_pnl,
            trades       = trades,
            sortino      = sortino,
            total_return = total_return,
            n_trades     = n_trades,
            win_rate     = win_rate,
            max_drawdown = 0.0,
            avg_hold_bars= avg_hold,
            profit_loss_ratio = pl_ratio,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 交易記錄提取
    # ─────────────────────────────────────────────────────────────────────

    def _extract_trades(
        self,
        symbol:      str,
        position:    np.ndarray,   # [T] 連續倉位
        open_prices: np.ndarray,   # [T]
        times:       np.ndarray,   # [T]
        pnl:         np.ndarray,   # [T]
    ) -> list[Trade]:
        """從倉位序列中提取完整交易列表（含開平倉 bar、價格、PnL）。

        執行價對齊邏輯（與 target_ret 計算保持一致）：
          target_ret[t] = log(open[t+2] / open[t+1])
          position[t] 產生的收益對應 open[t+1] → open[t+2]
          因此：信號在 entry_bar 產生 → 實際成交價 = open[entry_bar + 1]
                信號在 exit_bar 翻轉 → 實際成交價 = open[exit_bar + 1]

        PnL 計算：把持倉期間的逐 bar pnl 累加作為本筆盈虧。
        """
        T = len(position)
        trades:       list[Trade] = []
        cum_pnl_total = 0.0

        current_dir: int = 0
        entry_bar:   int = 0

        def _exec_price(bar: int) -> float:
            """信號在 bar 產生，執行價為下一根 open（若越界則取最後一根）。"""
            idx = min(bar + 1, T - 1)
            return float(open_prices[idx])

        def _exec_time(bar: int) -> int:
            idx = min(bar + 1, T - 1)
            return int(times[idx])

        for t in range(T):
            new_dir = target_to_direction(float(position[t]))

            if new_dir != current_dir:
                # 平掉舊倉
                if current_dir != 0:
                    trade_pnl = float(pnl[entry_bar:t].sum())
                    cum_pnl_total += trade_pnl
                    trade = Trade(
                        symbol      = symbol,
                        direction   = current_dir,
                        entry_bar   = entry_bar,
                        entry_time  = _exec_time(entry_bar),
                        entry_price = _exec_price(entry_bar),
                        exit_bar    = t,
                        exit_time   = _exec_time(t),
                        exit_price  = _exec_price(t),
                        pnl         = trade_pnl,
                        cum_pnl     = cum_pnl_total,
                    )
                    trades.append(trade)

                current_dir = new_dir
                entry_bar   = t

        # 序列末尾強平
        if current_dir != 0:
            trade_pnl = float(pnl[entry_bar:].sum())
            cum_pnl_total += trade_pnl
            trades.append(Trade(
                symbol      = symbol,
                direction   = current_dir,
                entry_bar   = entry_bar,
                entry_time  = _exec_time(entry_bar),
                entry_price = _exec_price(entry_bar),
                exit_bar    = T - 1,
                exit_time   = _exec_time(T - 1),
                exit_price  = _exec_price(T - 1),
                pnl         = trade_pnl,
                cum_pnl     = cum_pnl_total,
            ))

        return trades

    # ─────────────────────────────────────────────────────────────────────
    # 統計輔助
    # ─────────────────────────────────────────────────────────────────────

    def _calc_sortino(self, pnl: np.ndarray) -> float:
        mean_pnl = float(np.mean(pnl))
        downside = pnl[pnl < 0]
        if len(downside) == 0:
            return 0.0
        ds_std = float(np.std(downside, ddof=0))
        floor  = max(abs(mean_pnl), 1e-8)
        ds_std = max(ds_std, floor)
        sortino = mean_pnl / ds_std * math.sqrt(self.periods_per_year)
        return float(np.clip(sortino, -20.0, 20.0))

    @staticmethod
    def _calc_profit_loss_ratio(trades: list[Trade]) -> float | None:
        """盈虧比 = 平均盈利 / 平均虧損絕對值。無盈利或無虧損時返回 None。"""
        wins = [t.pnl for t in trades if t.pnl is not None and t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl is not None and t.pnl < 0]
        if not wins or not losses:
            return None
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return None
        return float(avg_win / avg_loss)
