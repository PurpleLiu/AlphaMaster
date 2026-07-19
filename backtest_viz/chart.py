"""
backtest_viz/chart.py — K 線圖 + 出入場標註 + PnL 子圖

依賴 matplotlib（見 requirements.txt）。
在 matplotlib 無法顯示 GUI 時自動切換到 Agg 後端，直接保存為 PNG/HTML。
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── matplotlib 後端自動選擇 ─────────────────────────────────────────────
try:
    import matplotlib
    _DISPLAY = os.environ.get("DISPLAY") or os.name == "nt"   # Windows 有 GUI
    if not _DISPLAY:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from .engine import SymbolResult, Trade


def _ts_to_label(ts: int, fmt: str = "%m-%d %H:%M") -> str:
    """Unix 秒 → 可讀字串（UTC）"""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)
    except Exception:
        return str(ts)


class BacktestChart:
    """可視化回測圖表生成器。

    為每個品種生成一張包含以下子圖的綜合圖：
      ① K 線圖（蠟燭圖 + 出入場三角標記 + 倉位背景色）
      ② 因子 / 信號強度折線
      ③ 逐 bar PnL 直方圖
      ④ 累計 PnL 曲線

    另可為每筆交易生成獨立縮放圖（plot_trade_zoom），
    只展示入場前後局部K線，入場/出場價格一目了然。

    用法：
        chart = BacktestChart()
        chart.plot(result, save_path="out/XAUUSD.png")
        chart.plot_all_trade_zooms(result, output_dir="out/")
    """

    # 顏色方案
    _LONG_ENTRY_COLOR  = "#26a69a"   # 綠色：開多
    _SHORT_ENTRY_COLOR = "#ef5350"   # 紅色：開空
    _EXIT_COLOR        = "#ffa726"   # 橙色：平倉
    _LONG_BG           = "#e8f5e9"   # 淺綠背景：多頭持倉期
    _SHORT_BG          = "#ffebee"   # 淺紅背景：空頭持倉期

    def __init__(
        self,
        figsize:    tuple[int, int] = (22, 14),
        max_bars:   int             = 120,       # 全局圖最多顯示的 bar 數，120根更清晰
        dpi:        int             = 120,
    ):
        if not _MPL_OK:
            raise ImportError(
                "matplotlib 未安裝。請運行: pip install matplotlib"
            )
        self.figsize  = figsize
        self.max_bars = max_bars
        self.dpi      = dpi

    # ─────────────────────────────────────────────────────────────────────
    # 公開介面
    # ─────────────────────────────────────────────────────────────────────

    def plot(
        self,
        result:    SymbolResult,
        save_path: Optional[str] = None,
        show:      bool          = False,
        title_suffix: str        = "",
    ) -> Optional[str]:
        """為單個品種生成完整圖表。

        Args:
            result:       BacktestEngine.run() 返回的 SymbolResult。
            save_path:    保存路徑（.png / .svg）；None 則不保存。
            show:         是否調用 plt.show()（交互環境下使用）。
            title_suffix: 附加到標題的額外說明。

        Returns:
            實際保存路徑字串，未保存時返回 None。
        """
        T   = len(result.times)
        # 顯示最後 max_bars 個 bar
        start_idx = max(0, T - self.max_bars)
        sl = slice(start_idx, T)

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs  = GridSpec(
            4, 1, figure=fig,
            height_ratios=[4, 1.2, 1.2, 1.5],
            hspace=0.08,
        )

        ax_candle = fig.add_subplot(gs[0])
        ax_factor = fig.add_subplot(gs[1], sharex=ax_candle)
        ax_pnl    = fig.add_subplot(gs[2], sharex=ax_candle)
        ax_cum    = fig.add_subplot(gs[3], sharex=ax_candle)

        x = np.arange(T)[sl]

        # ── ① K 線圖 ─────────────────────────────────────────────────
        self._draw_candles(ax_candle, result, sl, x)
        self._draw_trade_markers(ax_candle, result, start_idx, T)
        self._draw_position_background(ax_candle, result, start_idx, T, x)
        self._draw_trade_labels(ax_candle, result, start_idx, T)

        ax_candle.set_ylabel("Price", fontsize=9)
        ax_candle.legend(
            handles=self._legend_handles(), loc="upper left", fontsize=8,
            framealpha=0.7,
        )
        ax_candle.grid(alpha=0.3)
        pl_txt = (
            f"{result.profit_loss_ratio:.3f}"
            if result.profit_loss_ratio is not None
            else "—"
        )
        ax_candle.set_title(
            f"{result.symbol}  |  "
            f"Sortino={result.sortino:.2f}  "
            f"TotalRet={result.total_return:.4f}  "
            f"Trades={result.n_trades}  "
            f"WinRate={result.win_rate:.1%}  "
            f"PLRatio={pl_txt}  "
            f"AvgHold={result.avg_hold_bars:.1f}bars"
            + (f"  |  {title_suffix}" if title_suffix else ""),
            fontsize=10, pad=6,
        )

        # ── ② 因子強度 ────────────────────────────────────────────────
        ax_factor.plot(x, result.signal[sl], color="#7e57c2", linewidth=0.8,
                       label="signal (tanh)")
        ax_factor.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax_factor.fill_between(x, result.signal[sl], 0,
                               where=result.signal[sl] > 0,
                               alpha=0.15, color=self._LONG_ENTRY_COLOR)
        ax_factor.fill_between(x, result.signal[sl], 0,
                               where=result.signal[sl] < 0,
                               alpha=0.15, color=self._SHORT_ENTRY_COLOR)
        ax_factor.set_ylabel("Signal", fontsize=8)
        ax_factor.legend(fontsize=7, loc="upper left")
        ax_factor.grid(alpha=0.25)

        # ── ③ 逐 bar PnL 直方圖 ──────────────────────────────────────
        pnl_sl = result.pnl[sl]
        colors_bar = [
            self._LONG_ENTRY_COLOR if v >= 0 else self._SHORT_ENTRY_COLOR
            for v in pnl_sl
        ]
        ax_pnl.bar(x, pnl_sl, color=colors_bar, width=0.8, alpha=0.7)
        ax_pnl.axhline(0, color="gray", linewidth=0.6)
        ax_pnl.set_ylabel("Bar PnL", fontsize=8)
        ax_pnl.grid(alpha=0.25)

        # ── ④ 累計 PnL 曲線 ──────────────────────────────────────────
        cum_sl = result.cum_pnl[sl]
        ax_cum.plot(x, cum_sl, color="#1565c0", linewidth=1.2, label="Cum PnL")
        ax_cum.fill_between(x, cum_sl, 0,
                            where=cum_sl >= 0, alpha=0.12,
                            color=self._LONG_ENTRY_COLOR)
        ax_cum.fill_between(x, cum_sl, 0,
                            where=cum_sl < 0, alpha=0.12,
                            color=self._SHORT_ENTRY_COLOR)
        ax_cum.axhline(0, color="gray", linewidth=0.6)
        ax_cum.set_ylabel("Cum PnL", fontsize=8)
        ax_cum.legend(fontsize=7, loc="upper left")
        ax_cum.grid(alpha=0.25)

        # ── X 軸刻度（時間標籤）─────────────────────────────────────
        self._set_time_ticks(ax_cum, result.times[sl], x, n_ticks=10)
        plt.setp(ax_candle.get_xticklabels(), visible=False)
        plt.setp(ax_factor.get_xticklabels(), visible=False)
        plt.setp(ax_pnl.get_xticklabels(), visible=False)
        ax_cum.tick_params(axis="x", labelsize=7, rotation=30)

        # ── 保存 / 展示 ──────────────────────────────────────────────
        saved_path: Optional[str] = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            saved_path = save_path
            print(f"  圖表已保存 → {save_path}")

        if show:
            plt.show()

        plt.close(fig)
        return saved_path

    def plot_all(
        self,
        results:    list[SymbolResult],
        output_dir: str  = "backtest_output",
        show:       bool = False,
        title_suffix: str = "",
    ) -> list[str]:
        """為所有品種批次生成圖表。

        Returns:
            已保存的文件路徑列表。
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved = []
        for r in results:
            path = str(Path(output_dir) / f"{r.symbol}.png")
            p = self.plot(r, save_path=path, show=show,
                          title_suffix=title_suffix)
            if p:
                saved.append(p)
        return saved

    def plot_trade_zoom(
        self,
        result:       SymbolResult,
        trade_idx:    int,
        pre_bars:     int            = 20,   # 入場前顯示的K線數
        post_bars:    int            = 10,   # 出場後顯示的K線數
        save_path:    Optional[str]  = None,
        show:         bool           = False,
    ) -> Optional[str]:
        """為單筆交易生成局部縮放K線圖。

        只顯示入場前 pre_bars 根 + 持倉期 + 出場後 post_bars 根，
        清晰標註入場/出場價格水平線和具體價格數值。

        Args:
            result:    SymbolResult 回測結果。
            trade_idx: result.trades 中的交易索引（0-based）。
            pre_bars:  入場前顯示的K線數量，默認20根。
            post_bars: 出場後顯示的K線數量，默認10根。
            save_path: 保存路徑；None 則不保存。
            show:      是否調用 plt.show()。

        Returns:
            實際保存路徑，未保存時返回 None。
        """
        if trade_idx >= len(result.trades):
            raise IndexError(
                f"trade_idx={trade_idx} 超出範圍（共 {len(result.trades)} 筆交易）"
            )

        trade = result.trades[trade_idx]
        T     = len(result.times)

        # 實際成交在信號 bar 的下一根（與 target_ret 時間對齊）
        entry_exec_bar = min(trade.entry_bar + 1, T - 1)
        exit_exec_bar  = min(trade.exit_bar + 1, T - 1) if trade.exit_bar is not None else None

        # 計算顯示窗口（圍繞實際成交 bar 展開）
        win_start = max(0, entry_exec_bar - pre_bars)
        ref_end   = exit_exec_bar if exit_exec_bar is not None else entry_exec_bar
        win_end   = min(T, ref_end + post_bars + 1)
        sl  = slice(win_start, win_end)
        x   = np.arange(win_end - win_start)

        # 入場/出場在 x 軸的位置（用實際成交 bar）
        entry_xi = entry_exec_bar - win_start
        exit_xi  = (exit_exec_bar - win_start) if exit_exec_bar is not None else None

        fig, (ax_candle, ax_cum) = plt.subplots(
            2, 1, figsize=(14, 8), dpi=self.dpi,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        # ── K 線圖 ───────────────────────────────────────────────────
        self._draw_candles(ax_candle, result, sl, x)
        self._draw_position_background(ax_candle, result, win_start, win_end, x)

        # 入場標記（三角）
        lo = result.low[sl]
        hi = result.high[sl]
        cl = result.close[sl]
        offset = (hi.max() - lo.min()) * 0.008

        if trade.direction == 1:
            ax_candle.plot(
                entry_xi, lo[entry_xi] - offset,
                marker="^", color=self._LONG_ENTRY_COLOR,
                markersize=14, zorder=6, markeredgewidth=1,
                markeredgecolor="white",
            )
        else:
            ax_candle.plot(
                entry_xi, hi[entry_xi] + offset,
                marker="v", color=self._SHORT_ENTRY_COLOR,
                markersize=14, zorder=6, markeredgewidth=1,
                markeredgecolor="white",
            )

        # 出場標記（菱形）
        if exit_xi is not None:
            ax_candle.plot(
                exit_xi, cl[exit_xi],
                marker="D", color=self._EXIT_COLOR,
                markersize=10, zorder=6, markeredgewidth=1,
                markeredgecolor="white",
            )

        # 入場價格水平虛線
        entry_price = trade.entry_price
        ax_candle.axhline(
            entry_price, color=self._LONG_ENTRY_COLOR if trade.direction == 1
            else self._SHORT_ENTRY_COLOR,
            linewidth=1.2, linestyle="--", alpha=0.8, zorder=4,
        )
        ax_candle.annotate(
            f"Entry  {entry_price:.5f}",
            xy=(x[-1], entry_price),
            xytext=(-4, 4), textcoords="offset points",
            fontsize=8, color=self._LONG_ENTRY_COLOR if trade.direction == 1
            else self._SHORT_ENTRY_COLOR,
            ha="right", fontweight="bold",
        )

        # 出場價格水平虛線
        if trade.exit_price is not None and exit_xi is not None:
            ax_candle.axhline(
                trade.exit_price, color=self._EXIT_COLOR,
                linewidth=1.2, linestyle="--", alpha=0.8, zorder=4,
            )
            ax_candle.annotate(
                f"Exit  {trade.exit_price:.5f}",
                xy=(x[-1], trade.exit_price),
                xytext=(-4, -8), textcoords="offset points",
                fontsize=8, color=self._EXIT_COLOR,
                ha="right", fontweight="bold",
            )
            # 入場→出場垂直價差連線
            ax_candle.annotate(
                "",
                xy=(exit_xi, trade.exit_price),
                xytext=(entry_xi, entry_price),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#9e9e9e",
                    lw=1.2,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=3,
            )

        # PnL 標註框
        pnl_color = "#1b5e20" if trade.pnl > 0 else "#b71c1c"
        direction_str = "Long ▲" if trade.direction == 1 else "Short ▼"
        hold_bars = (trade.exit_bar - trade.entry_bar) if trade.exit_bar is not None else 0
        entry_time_str = _ts_to_label(trade.entry_time, "%Y-%m-%d %H:%M")
        exit_time_str  = (
            _ts_to_label(trade.exit_time, "%Y-%m-%d %H:%M")
            if trade.exit_time else "open"
        )
        exit_price_str = f"{trade.exit_price:.5f}" if trade.exit_price else "-"
        info_text = (
            f"{direction_str}  PnL: {trade.pnl:+.5f}\n"
            f"Entry: {entry_time_str} @ {entry_price:.5f}\n"
            f"Exit : {exit_time_str} @ {exit_price_str}\n"
            f"Hold : {hold_bars} bars"
        )
        ax_candle.text(
            0.01, 0.97, info_text,
            transform=ax_candle.transAxes,
            fontsize=8.5, verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white", alpha=0.85,
                edgecolor=pnl_color, linewidth=1.5,
            ),
            color=pnl_color,
        )

        # 標題
        trade_no = trade_idx + 1
        ax_candle.set_title(
            f"{result.symbol}  Trade {trade_no}/{result.n_trades}  "
            f"{'Long' if trade.direction == 1 else 'Short'}  "
            f"PnL={trade.pnl:+.5f}  WinRate={result.win_rate:.1%}",
            fontsize=10, pad=6,
        )
        ax_candle.set_ylabel("Price", fontsize=9)
        ax_candle.grid(alpha=0.3)
        ax_candle.legend(
            handles=self._legend_handles(), loc="upper right",
            fontsize=7, framealpha=0.7,
        )

        # ── 累計 PnL 曲線（全局，標註當前交易位置）─────────────────
        cum_all = result.cum_pnl
        x_all   = np.arange(T)
        ax_cum.plot(x_all, cum_all, color="#1565c0", linewidth=1.0, label="Cum PnL")
        ax_cum.fill_between(
            x_all, cum_all, 0,
            where=cum_all >= 0, alpha=0.10, color=self._LONG_ENTRY_COLOR,
        )
        ax_cum.fill_between(
            x_all, cum_all, 0,
            where=cum_all < 0, alpha=0.10, color=self._SHORT_ENTRY_COLOR,
        )
        # 標註當前交易在全局 PnL 上的位置
        if trade.exit_bar is not None:
            ax_cum.axvspan(
                trade.entry_bar, trade.exit_bar,
                alpha=0.25,
                color=self._LONG_BG if trade.direction == 1 else self._SHORT_BG,
            )
        ax_cum.axhline(0, color="gray", linewidth=0.6)
        ax_cum.set_ylabel("Cum PnL", fontsize=8)
        ax_cum.legend(fontsize=7, loc="upper left")
        ax_cum.grid(alpha=0.25)
        self._set_time_ticks(ax_cum, result.times, x_all, n_ticks=8)
        ax_cum.tick_params(axis="x", labelsize=7, rotation=30)

        # X 軸時間刻度（局部 K 線圖）
        self._set_time_ticks(ax_candle, result.times[sl], x, n_ticks=6)
        ax_candle.tick_params(axis="x", labelsize=7, rotation=20)

        # 保存 / 展示
        saved_path: Optional[str] = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            saved_path = save_path
            print(f"  縮放圖已保存 → {save_path}")

        if show:
            plt.show()

        plt.close(fig)
        return saved_path

    def plot_all_trade_zooms(
        self,
        result:       SymbolResult,
        output_dir:   str  = "backtest_output",
        pre_bars:     int  = 20,
        post_bars:    int  = 10,
        max_trades:   int  = 30,    # 最多生成多少張縮放圖（避免文件爆炸）
        show:         bool = False,
    ) -> list[str]:
        """為一個品種的所有交易批次生成縮放圖。

        Args:
            result:     SymbolResult 回測結果。
            output_dir: 輸出目錄。
            pre_bars:   入場前顯示K線數量。
            post_bars:  出場後顯示K線數量。
            max_trades: 最多生成張數（按 |PnL| 降序選 top-N）。
            show:       是否調用 plt.show()。

        Returns:
            已保存的文件路徑列表。
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved = []

        trades = result.trades
        if not trades:
            print(f"  {result.symbol} 無交易記錄，跳過縮放圖生成。")
            return saved

        # 選取最重要的 max_trades 筆（按 |PnL| 降序）
        indexed = sorted(
            enumerate(trades),
            key=lambda kv: abs(kv[1].pnl),
            reverse=True,
        )[:max_trades]
        # 按時間順序重新排序，編號更直觀
        indexed.sort(key=lambda kv: kv[1].entry_bar)

        for rank, (orig_idx, _) in enumerate(indexed, start=1):
            fname = f"{result.symbol}_trade{rank}_zoom.png"
            path  = str(Path(output_dir) / fname)
            p = self.plot_trade_zoom(
                result, orig_idx,
                pre_bars=pre_bars, post_bars=post_bars,
                save_path=path, show=show,
            )
            if p:
                saved.append(p)

        return saved

    # ─────────────────────────────────────────────────────────────────────
    # 內部繪圖方法
    # ─────────────────────────────────────────────────────────────────────

    def _draw_candles(
        self,
        ax:     "plt.Axes",
        result: SymbolResult,
        sl:     slice,
        x:      np.ndarray,
    ) -> None:
        """繪製蠟燭圖（用矩形 + 線段模擬，不依賴 mplfinance）"""
        op = result.open[sl]
        hi = result.high[sl]
        lo = result.low[sl]
        cl = result.close[sl]

        bar_w = 0.6
        for i, xi in enumerate(x):
            is_bull = cl[i] >= op[i]
            color   = self._LONG_ENTRY_COLOR if is_bull else self._SHORT_ENTRY_COLOR
            # 上下影線
            ax.plot([xi, xi], [lo[i], hi[i]], color=color, linewidth=0.7)
            # 實體
            body_lo = min(op[i], cl[i])
            body_hi = max(op[i], cl[i])
            body_h  = max(body_hi - body_lo, (hi[i] - lo[i]) * 0.01)
            rect = mpatches.Rectangle(
                (xi - bar_w / 2, body_lo), bar_w, body_h,
                linewidth=0, facecolor=color, alpha=0.85,
            )
            ax.add_patch(rect)

        ax.set_xlim(x[0] - 1, x[-1] + 1)
        ax.set_ylim(result.low[sl].min() * 0.9995,
                    result.high[sl].max() * 1.0005)

    def _draw_position_background(
        self,
        ax:        "plt.Axes",
        result:    SymbolResult,
        start_idx: int,
        T:         int,
        x:         np.ndarray,
    ) -> None:
        """在持倉期間填充背景色（多頭綠/空頭紅）。

        背景從實際成交 bar 開始（信號 bar + 1）。
        """
        for trade in result.trades:
            entry_exec = min(trade.entry_bar + 1, T - 1)
            exit_exec  = min(trade.exit_bar + 1, T - 1) if trade.exit_bar is not None else T - 1

            # 轉換為 x 軸坐標（相對於 start_idx）
            x_entry = max(entry_exec - start_idx, 0)
            x_exit  = min(exit_exec  - start_idx, len(x) - 1)

            if x_entry >= len(x) or x_exit < 0:
                continue

            color = self._LONG_BG if trade.direction == 1 else self._SHORT_BG
            ax.axvspan(x[x_entry], x[x_exit], alpha=0.25, color=color, linewidth=0)

    def _draw_trade_markers(
        self,
        ax:        "plt.Axes",
        result:    SymbolResult,
        start_idx: int,
        T:         int,
    ) -> None:
        """繪製開平倉三角標記。

        標記打在實際成交 bar（信號 bar + 1），與 entry_price/exit_price 對齊。

        多頭開倉：綠色向上三角（▲），標註在 low 下方
        空頭開倉：紅色向下三角（▼），標註在 high 上方
        平倉/反手：橙色菱形（◆），標註在 close 附近
        """
        lo = result.low
        hi = result.high
        cl = result.close
        total = T

        for trade in result.trades:
            # 實際成交 bar = 信號 bar + 1
            eb = min(trade.entry_bar + 1, total - 1)
            xb = min(trade.exit_bar + 1, total - 1) if trade.exit_bar is not None else None

            if eb >= start_idx:
                xi = eb - start_idx
                offset = (hi[eb] - lo[eb]) * 0.3 + (hi[eb] - lo[eb]) * 0.05
                if trade.direction == 1:
                    ax.plot(xi, lo[eb] - offset,
                            marker="^", color=self._LONG_ENTRY_COLOR,
                            markersize=8, zorder=5, markeredgewidth=0.5,
                            markeredgecolor="white")
                else:
                    ax.plot(xi, hi[eb] + offset,
                            marker="v", color=self._SHORT_ENTRY_COLOR,
                            markersize=8, zorder=5, markeredgewidth=0.5,
                            markeredgecolor="white")

            if xb is not None and xb >= start_idx:
                xi = xb - start_idx
                ax.plot(xi, cl[xb],
                        marker="D", color=self._EXIT_COLOR,
                        markersize=6, zorder=5, markeredgewidth=0.5,
                        markeredgecolor="white")

    def _draw_trade_labels(
        self,
        ax:        "plt.Axes",
        result:    SymbolResult,
        start_idx: int,
        T:         int,
    ) -> None:
        """在每筆交易標註 PnL 數值（僅盈虧超過閾值時顯示，避免文字過密）。"""
        hi = result.high
        lo = result.low
        price_range = hi.max() - lo.min()
        threshold   = price_range * 0.001

        visible = [
            t for t in result.trades
            if abs(t.pnl) > threshold
            and t.exit_bar is not None
            and min(t.exit_bar + 1, T - 1) >= start_idx
        ]
        if len(visible) > 20:
            visible = sorted(visible, key=lambda t: abs(t.pnl), reverse=True)[:20]

        for trade in visible:
            if trade.exit_bar is None:
                continue
            # 標註在實際成交（出場）的 bar 上
            xb = min(trade.exit_bar + 1, T - 1)
            xi    = xb - start_idx
            price = result.close[xb]
            label = f"{trade.pnl:+.4f}"
            color = "#1b5e20" if trade.pnl > 0 else "#b71c1c"
            ax.annotate(
                label,
                xy=(xi, price),
                xytext=(0, 12 if trade.pnl > 0 else -16),
                textcoords="offset points",
                fontsize=6, color=color,
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.1", fc="white",
                    alpha=0.6, edgecolor="none",
                ),
            )

    @staticmethod
    def _set_time_ticks(
        ax:      "plt.Axes",
        times:   np.ndarray,
        x:       np.ndarray,
        n_ticks: int = 10,
    ) -> None:
        """設置 X 軸時間刻度標籤。"""
        step  = max(1, len(x) // n_ticks)
        ticks = x[::step]
        labels = [_ts_to_label(int(times[i]), fmt="%y-%m-%d\n%H:%M")
                  for i in range(0, len(times), step)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels[:len(ticks)])

    def _legend_handles(self) -> list:
        """構建圖例 handles"""
        return [
            mpatches.Patch(color=self._LONG_ENTRY_COLOR, label="▲ Long entry"),
            mpatches.Patch(color=self._SHORT_ENTRY_COLOR, label="▼ Short entry"),
            mpatches.Patch(color=self._EXIT_COLOR,        label="◆ Exit"),
            mpatches.Patch(color=self._LONG_BG,  alpha=0.4, label="Long position"),
            mpatches.Patch(color=self._SHORT_BG, alpha=0.4, label="Short position"),
        ]
