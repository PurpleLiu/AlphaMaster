"""
run_backtest.py — 多因子組合回測（含手續費/滑點、夏普、資金曲線）

訓練/回測一律使用本地 Parquet，不連接 MT5 在線拉數。

用法：
    python run_backtest.py --strategy-file strategies/best_ADAUSD.json --data-file D:\\K線數據\\ADAUSD_H1.parquet
    python run_backtest.py --strategy-file path\\to\\strategy.json
        # 若策略 JSON 內含 data_file 欄位，可省略 --data-file
    python run_backtest.py --commission 0.02 --slippage 0.01
        # 單邊手續費/滑點（單位 %），默認 0.02 / 0.01
"""

import json, sys, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager
from backtest_viz import BacktestEngine
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from model_core.backtest import estimate_periods_per_year
from strategy_manager.signal import compute_target_positions_stateless

_H1_PER_YEAR = 6240
DEFAULT_COMMISSION_PCT = 0.02  # 單邊手續費 %
DEFAULT_SLIPPAGE_PCT = 0.01    # 單邊滑點 %


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def load_strategy(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"formula": data, "vocab_version": "legacy", "symbol": None}
    return data


# ── 統計指標 ──────────────────────────────────────────────────────────────────

def calc_sharpe(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sharpe（無風險利率=0）。"""
    m = pnl.mean()
    s = pnl.std(ddof=0)
    if s < 1e-10:
        return 0.0
    return float(m / s * math.sqrt(periods_per_year))


def calc_sortino(pnl: np.ndarray, periods_per_year: int = _H1_PER_YEAR) -> float:
    """年化 Sortino（下行標準差）。"""
    m    = pnl.mean()
    down = pnl[pnl < 0]
    ds   = down.std(ddof=0) if len(down) > 0 else 1e-10
    ds   = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(periods_per_year), -20, 20))


def calc_rolling_sharpe(
    pnl: np.ndarray,
    window: int = 500,
    periods_per_year: int = _H1_PER_YEAR,
) -> np.ndarray:
    """滾動年化夏普；窗口不足處為 nan。"""
    T = len(pnl)
    out = np.full(T, np.nan, dtype=np.float64)
    if T == 0 or window <= 1:
        return out
    w = min(window, T)
    # 累積和 / 累積平方和 → O(T) 滑動窗口
    csum = np.concatenate([[0.0], np.cumsum(pnl, dtype=np.float64)])
    csq = np.concatenate([[0.0], np.cumsum(pnl.astype(np.float64) ** 2)])
    for i in range(w - 1, T):
        s = csum[i + 1] - csum[i + 1 - w]
        sq = csq[i + 1] - csq[i + 1 - w]
        mean = s / w
        var = sq / w - mean * mean
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.clip(mean / std * math.sqrt(periods_per_year), -20, 20))
    return out


def _fmt_pl_ratio(results_map: dict) -> str:
    vals = [
        d["profit_loss_ratio"]
        for d in results_map.values()
        if d.get("profit_loss_ratio") is not None
    ]
    if not vals:
        return "—"
    return f"{sum(vals) / len(vals):.3f}"


# ── 資金曲線圖 ────────────────────────────────────────────────────────────────

def _setup_chinese_font() -> None:
    """讓 matplotlib 能正確顯示中文（Windows 優先微軟雅黑）。"""
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_equity_curves(results_map: dict, output_dir: str, times_arr: np.ndarray | None = None,
                       periods_per_year: int = _H1_PER_YEAR):
    """繪製各品種 + 等權組合的資金曲線（中文標註）。

    Args:
        results_map: {symbol: {"pnl": np.array, "cum_pnl": np.array, ...}}
        output_dir:  輸出目錄
        times_arr:   時間戳數組（Unix秒），用於 X 軸刻度
        periods_per_year: 年化因子（每年 bar 數），由調用方按數據週期估計
    """
    _setup_chinese_font()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    syms   = list(results_map.keys())
    n_syms = len(syms)

    fig, ax_eq = plt.subplots(figsize=(18, 7), dpi=110)

    colors = ["#1565c0", "#00897b", "#e65100", "#6a1b9a", "#558b2f", "#b71c1c"]

    # 等權組合 PnL
    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)

    T = len(port_cum)
    x = np.arange(T)

    if n_syms == 1:
        sym = syms[0]
        cum = results_map[sym]["cum_pnl"]
        ax_eq.plot(
            x, cum, linewidth=2.0, color="#1565c0",
            label=f"{sym}（索提諾 {results_map[sym]['sortino']:+.2f}）",
        )
        ax_eq.fill_between(x, cum, 0, where=cum >= 0, alpha=0.08, color="#1565c0")
        ax_eq.fill_between(x, cum, 0, where=cum < 0,  alpha=0.08, color="#b71c1c")
        title_head = f"{sym} 資金曲線"
        show_pnl, show_cum = results_map[sym]["pnl"], cum
    else:
        for i, sym in enumerate(syms):
            cum = results_map[sym]["cum_pnl"]
            ax_eq.plot(
                x, cum, linewidth=0.8, alpha=0.65, color=colors[i % len(colors)],
                label=f"{sym}（索提諾 {results_map[sym]['sortino']:+.2f}）",
            )
        ax_eq.plot(
            x, port_cum, linewidth=2.2, color="black",
            label=f"等權組合（索提諾 {calc_sortino(port_pnl, periods_per_year):+.2f}）",
        )
        ax_eq.fill_between(x, port_cum, 0, where=port_cum >= 0, alpha=0.06, color="#1565c0")
        ax_eq.fill_between(x, port_cum, 0, where=port_cum < 0,  alpha=0.06, color="#b71c1c")
        title_head = "多因子組合資金曲線"
        show_pnl, show_cum = port_pnl, port_cum

    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("累計對數收益", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(
        f"{title_head}  |  "
        f"總收益={show_cum[-1]:+.3f}  "
        f"夏普={calc_sharpe(show_pnl, periods_per_year):+.2f}  "
        f"索提諾={calc_sortino(show_pnl, periods_per_year):+.2f}  "
        f"盈虧比={_fmt_pl_ratio(results_map)}",
        fontsize=11, pad=8,
    )

    # X 軸時間刻度
    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone
        step  = max(1, T // 10)
        ticks = x[::step]
        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%Y-%m-%d")
            for i in range(0, T, step)
        ]
        ax_eq.set_xticks(ticks)
        ax_eq.set_xticklabels(labels[:len(ticks)], fontsize=8, rotation=20)
    ax_eq.set_xlabel("日期", fontsize=9)

    path = str(Path(output_dir) / "portfolio_equity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  資金曲線圖已保存 → {path}")
    return path


def export_equity_json(
    results_map: dict,
    output_dir: str,
    times_arr: np.ndarray | None = None,
    max_points: int = 1500,
    rolling_window: int = 500,
    periods_per_year: int = _H1_PER_YEAR,
):
    """導出資金曲線原始數據為 JSON，供前端渲染互動式 HTML 圖表。

    結構：
        {
          "labels": [...時間標籤],
          "n_points": int, "total_bars": int,
          "rolling_window": int,
          "symbols": { sym: { equity, rolling_sharpe, sharpe, sortino,
                              total_return, profit_loss_ratio } },
          "portfolio": { ... }   # 多品種時才有
        }
    """
    syms = list(results_map.keys())
    if not syms:
        return None

    all_pnls = np.stack([results_map[s]["pnl"] for s in syms], axis=0)
    port_pnl = all_pnls.mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    T = len(port_cum)

    # 均勻降採樣，保證首尾點在內，避免 JSON 過大導致前端卡頓
    if T > max_points:
        idx = np.unique(np.linspace(0, T - 1, max_points).astype(int))
    else:
        idx = np.arange(T)

    def _sample(arr: np.ndarray) -> list[float | None]:
        out = []
        for i in idx:
            v = arr[i]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(None)
            else:
                out.append(round(float(v), 6))
        return out

    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone

        labels = [
            datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            for i in idx
        ]
    else:
        labels = [str(int(i)) for i in idx]

    out: dict = {
        "labels": labels,
        "n_points": int(len(idx)),
        "total_bars": int(T),
        "rolling_window": int(rolling_window),
        "symbols": {},
    }
    for s in syms:
        cum = results_map[s]["cum_pnl"]
        roll = calc_rolling_sharpe(results_map[s]["pnl"], window=rolling_window,
                                   periods_per_year=periods_per_year)
        pl = results_map[s].get("profit_loss_ratio")
        out["symbols"][s] = {
            "equity": _sample(cum),
            "rolling_sharpe": _sample(roll),
            "sharpe": round(float(results_map[s]["sharpe"]), 4),
            "sortino": round(float(results_map[s]["sortino"]), 4),
            "total_return": round(float(results_map[s]["total_return"]), 6),
            "profit_loss_ratio": round(float(pl), 4) if pl is not None else None,
        }

    if len(syms) > 1:
        pl_vals = [
            results_map[s]["profit_loss_ratio"]
            for s in syms
            if results_map[s].get("profit_loss_ratio") is not None
        ]
        port_pl = float(sum(pl_vals) / len(pl_vals)) if pl_vals else None
        out["portfolio"] = {
            "equity": _sample(port_cum),
            "rolling_sharpe": _sample(calc_rolling_sharpe(port_pnl, window=rolling_window,
                                                         periods_per_year=periods_per_year)),
            "sharpe": round(float(calc_sharpe(port_pnl, periods_per_year)), 4),
            "sortino": round(float(calc_sortino(port_pnl, periods_per_year)), 4),
            "total_return": round(float(port_cum[-1]), 6),
            "profit_loss_ratio": round(port_pl, 4) if port_pl is not None else None,
        }

    path = Path(output_dir) / "equity_curve.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  資金曲線數據已保存 → {path}")
    return str(path)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR  = "backtest_output"
    single_mode = "--single" in sys.argv
    # 回測強制離線：只用本地 Parquet，永不連 MT5 在線上
    if "--online" in sys.argv or "--mt5" in sys.argv:
        print("[ERROR] 回測已禁用在線/MT5 拉數。請使用本地 Parquet（--data-file 或策略內 data_file）。")
        sys.exit(1)

    strategy_file = None
    data_file_arg = None
    commission_pct = DEFAULT_COMMISSION_PCT
    slippage_pct = DEFAULT_SLIPPAGE_PCT
    for i, arg in enumerate(sys.argv):
        if arg == "--strategy-file" and i + 1 < len(sys.argv):
            strategy_file = sys.argv[i + 1]
        elif arg == "--data-file" and i + 1 < len(sys.argv):
            data_file_arg = sys.argv[i + 1]
        elif arg == "--commission" and i + 1 < len(sys.argv):
            commission_pct = float(sys.argv[i + 1])
        elif arg == "--slippage" and i + 1 < len(sys.argv):
            slippage_pct = float(sys.argv[i + 1])

    if commission_pct < 0 or slippage_pct < 0:
        print("[ERROR] 手續費/滑點不能為負"); sys.exit(1)
    cost_rate_all = (commission_pct + slippage_pct) / 100.0
    print(
        f"\n交易成本（單邊）: "
        f"手續費={commission_pct:g}%  滑點={slippage_pct:g}%  "
        f"→ cost_rate={cost_rate_all:.8f}"
    )
    print("數據模式: 強制離線 Parquet（不連接 MT5）")

    # ── 2. 載入策略 ─────────────────────────────────────────────────
    strategy_data_file = None
    print(f"\n{'='*62}")
    if strategy_file:
        data = load_strategy(Path(strategy_file))
        if data is None:
            print(f"[ERROR] 找不到: {strategy_file}"); sys.exit(1)
        strategy_data_file = data.get("data_file")
        sym = data.get("symbol")
        if not sym:
            stem = Path(strategy_file).stem
            if stem.startswith("best_"):
                sym = stem.replace("best_", "", 1)
            elif stem.startswith("strategy_"):
                # strategy_ADAUSD_step0084_score2.4021 / strategy_ADAUSD (1)
                rest = stem.replace("strategy_", "", 1)
                sym = rest.split("_step")[0].split(" ")[0]
        if not sym:
            print("[ERROR] 策略文件未包含 symbol，且無法從檔案名識別"); sys.exit(1)
        symbol_formulas = {sym: data["formula"]}
        sc = data.get("best_score", "N/A")
        score_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) else str(sc)
        print(f"  模式: 單策略文件 ({Path(strategy_file).name})")
        print(f"  {sym}: score={score_txt}  {decode_formula(data['formula'])}")
        if strategy_data_file:
            print(f"  策略記錄數據: {strategy_data_file}")
    elif single_mode:
        data = load_strategy(Path(Config.STRATEGY_FILE))
        if data is None:
            print(f"[ERROR] 找不到: {Config.STRATEGY_FILE}"); sys.exit(1)
        strategy_data_file = data.get("data_file")
        symbol_formulas = {sym: data["formula"] for sym in Config.SYMBOLS}
        print("  模式: 單公式（所有品種共用）")
    else:
        symbol_formulas = {}
        for sym in Config.SYMBOLS:
            path = Path("strategies") / f"best_{sym}.json"
            data = load_strategy(path)
            if data is None:
                print(f"  [缺失] {sym}")
                continue
            ver = data.get("vocab_version", "unknown")
            if ver != VOCAB_VERSION:
                print(f"  [跳過] {sym}: vocab_version 不符 ({ver} vs {VOCAB_VERSION})")
                continue
            symbol_formulas[sym] = data["formula"]
            if not strategy_data_file and data.get("data_file"):
                strategy_data_file = data.get("data_file")
            sc = data.get("best_score", "N/A")
            print(f"  {sym}: score={sc:.3f}  {decode_formula(data['formula'])}")

    if not symbol_formulas:
        print("[ERROR] 沒有有效策略，請先運行訓練"); sys.exit(1)

    cost_rates = {sym: cost_rate_all for sym in symbol_formulas}
    print(f"{'='*62}\n")

    # ── 3. 載入數據（僅本地 Parquet）────────────────────────────────
    if not data_file_arg and strategy_data_file:
        data_file_arg = str(strategy_data_file).strip() or None

    if not data_file_arg:
        print(
            "[ERROR] 未指定本地 Parquet。\n"
            "請傳入 --data-file PATH\\TO\\SYMBOL_TF.parquet，\n"
            "或使用包含 data_file 欄位的策略 JSON（本軟體訓練生成）。\n"
            "回測不會連接 MT5 / 不會使用在線行情。"
        )
        sys.exit(1)

    parquet_path = Path(data_file_arg)
    if not parquet_path.exists():
        print(f"[ERROR] Parquet 不存在: {parquet_path}")
        sys.exit(1)

    print(f"正在載入數據（離線 Parquet: {parquet_path}）...")
    pm = ParquetDataManager(str(parquet_path))
    pm.load()
    raw_dict = pm.raw_dict
    syms = pm.symbols
    # 策略品種名與 Parquet 品種不一致時（單策略 + 單品種文件），映射公式到數據品種
    if strategy_file and len(symbol_formulas) == 1 and len(syms) == 1:
        strat_sym = next(iter(symbol_formulas))
        data_sym = syms[0]
        if strat_sym != data_sym:
            formula = symbol_formulas[strat_sym]
            print(f"  [映射] 策略品種 {strat_sym} → 數據品種 {data_sym}")
            symbol_formulas = {data_sym: formula}
            cost_rates = {data_sym: cost_rate_all}

    T = raw_dict["open"].shape[1]
    times_all = raw_dict.get("time", None)
    # 數據驅動年化因子：按實際時間戳估計每年 bar 數，替代寫死的 H1=6240。
    # A 股日線(~244)、A 股 15min(~3904)、外匯 H1(6240)、加密日線(365) 都會被正確年化。
    ppy = estimate_periods_per_year(times_all) if times_all is not None else _H1_PER_YEAR
    print(f"  品種: {syms}  T={T} bars  年化因子={ppy} bar/年\n")

    # ── 4. 為每品種計算因子 + 回測 ───────────────────────────────
    vm   = StackVM()
    # 因果特徵化：_robust_norm 現為滾動窗口實現，傳入全量序列是安全的
    # 每個時間步 t 的歸一化參數只依賴 [t-w+1..t]，無 look-ahead
    feat = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]，因果安全

    results_map = {}
    backtest_results = []

    for i, sym in enumerate(syms):
        if sym not in symbol_formulas:
            print(f"  [跳過] {sym}（無策略）")
            continue

        formula   = symbol_formulas[sym]
        cost_rate = cost_rates.get(sym, cost_rate_all)
        feat_i    = feat[i:i+1]
        raw_i     = {k: v[i:i+1] for k, v in raw_dict.items()}

        engine    = BacktestEngine(formula=formula, cost_rate=cost_rate,
                                   periods_per_year=ppy)
        sym_res   = engine.run(raw_i, feat_i, [sym])
        backtest_results.extend(sym_res)

        r = sym_res[0]
        pnl_arr = r.pnl
        cum_arr = r.cum_pnl
        sharpe  = calc_sharpe(pnl_arr, ppy)
        sortino = calc_sortino(pnl_arr, ppy)
        pl_ratio = r.profit_loss_ratio

        results_map[sym] = {
            "pnl":          pnl_arr,
            "cum_pnl":      cum_arr,
            "total_return": r.total_return,
            "sharpe":       sharpe,
            "sortino":      sortino,
            "n_trades":     r.n_trades,
            "win_rate":     r.win_rate,
            "avg_hold":     r.avg_hold_bars,
            "profit_loss_ratio": pl_ratio,
            "cost_rate":    cost_rate,
        }

    # ── 5. 列印各品種統計 ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  多因子回測報告")
    print(f"{'='*62}")
    header = f"{'品種':12s} {'PnL':>8} {'Sharpe':>8} {'Sortino':>8} {'盈虧比':>8} {'Trades':>7} {'WinRate':>8} {'AvgH':>6}"
    print(f"  {header}")
    print(f"  {'─'*72}")
    for sym, d in results_map.items():
        pl = d["profit_loss_ratio"]
        pl_s = f"{pl:8.3f}" if pl is not None else f"{'—':>8}"
        print(f"  {sym:12s} "
              f"{d['total_return']:+8.3f} "
              f"{d['sharpe']:+8.3f} "
              f"{d['sortino']:+8.3f} "
              f"{pl_s} "
              f"{d['n_trades']:7d} "
              f"{d['win_rate']:8.1%} "
              f"{d['avg_hold']:6.1f}h")

    # 等權組合
    p_pl_ratio = None
    if results_map:
        all_pnls = np.stack([d["pnl"] for d in results_map.values()], axis=0)
        port_pnl = all_pnls.mean(axis=0)
        port_cum = np.cumsum(port_pnl)
        p_sharpe  = calc_sharpe(port_pnl, ppy)
        p_sortino = calc_sortino(port_pnl, ppy)
        pl_vals = [d["profit_loss_ratio"] for d in results_map.values()
                   if d["profit_loss_ratio"] is not None]
        p_pl_ratio = float(sum(pl_vals) / len(pl_vals)) if pl_vals else None
        pl_s = f"{p_pl_ratio:8.3f}" if p_pl_ratio is not None else f"{'—':>8}"
        print(f"  {'─'*72}")
        print(f"  {'Portfolio':12s} "
              f"{port_cum[-1]:+8.3f} "
              f"{p_sharpe:+8.3f} "
              f"{p_sortino:+8.3f} "
              f"{pl_s}")
        print(f"\n  正收益品種: {sum(1 for d in results_map.values() if d['total_return']>0)}/{len(results_map)}")
        print(f"  Sharpe>1 品種: {sum(1 for d in results_map.values() if d['sharpe']>1)}/{len(results_map)}")
    print(f"{'='*62}\n")

    # ── 6. 資金曲線圖 ─────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if results_map:
        times_np = times_all[0].numpy() if times_all is not None else None
        plot_equity_curves(results_map, OUTPUT_DIR, times_np, periods_per_year=ppy)
        export_equity_json(results_map, OUTPUT_DIR, times_np, periods_per_year=ppy)

    # ── 7. 資金曲線圖已在步驟 6 生成；跳過 K 線/逐筆交易圖以加快回測 ─────

    # ── 8. 保存 JSON 報告 ─────────────────────────────────────────────
    report = {
        "mode": "single" if single_mode else "multi_factor",
        "cost_rates": cost_rates,
        "symbols": {},
        "portfolio": {},
    }
    for sym, d in results_map.items():
        formula = symbol_formulas.get(sym, [])
        pl = d["profit_loss_ratio"]
        report["symbols"][sym] = {
            "formula":      formula,
            "readable":     decode_formula(formula),
            "cost_rate":    d["cost_rate"],
            "total_return": round(d["total_return"], 6),
            "sharpe":       round(d["sharpe"], 4),
            "sortino":      round(d["sortino"], 4),
            "n_trades":     d["n_trades"],
            "win_rate":     round(d["win_rate"], 4),
            "avg_hold_bars":round(d["avg_hold"], 2),
            "profit_loss_ratio": round(pl, 4) if pl is not None else None,
        }
    if results_map:
        report["portfolio"] = {
            "total_return": round(float(port_cum[-1]), 6),
            "sharpe":       round(p_sharpe, 4),
            "sortino":      round(p_sortino, 4),
            "profit_loss_ratio": round(p_pl_ratio, 4) if p_pl_ratio is not None else None,
        }
    rp = f"{OUTPUT_DIR}/multi_factor_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 報告已保存 → {rp}")
    print("完成。\n")


if __name__ == "__main__":
    main()
