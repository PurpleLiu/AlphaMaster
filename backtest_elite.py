"""
backtest_elite.py — 對 elite pool 中的最優因子批次回測，發現問題

用法：
    python backtest_elite.py --offline
"""
import sys, json, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

_H1_PER_YEAR = 6240


def decode(toks):
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in toks)


def calc_sharpe(pnl):
    m, s = pnl.mean(), pnl.std()
    return float(m / (s + 1e-10) * math.sqrt(_H1_PER_YEAR))


def calc_sortino(pnl):
    m = pnl.mean()
    down = pnl[pnl < 0]
    ds = down.std() if len(down) > 0 else 1e-10
    ds = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(_H1_PER_YEAR), -20, 20))


def calc_mdd(cum_pnl):
    peak = np.maximum.accumulate(cum_pnl)
    return float((peak - cum_pnl).max())


def calc_ic(factor, target_ret):
    """時序 IC：factor[t] vs target_ret[t+1]，逐品種再取均值。"""
    N, T = factor.shape
    ic_list = []
    for n in range(N):
        x = factor[n, :-1]
        y = target_ret[n, 1:]
        xm = x - x.mean()
        ym = y - y.mean()
        sx = np.sqrt((xm**2).mean())
        sy = np.sqrt((ym**2).mean())
        if sx < 1e-6 or sy < 1e-6:
            continue
        ic = (xm * ym).mean() / (sx * sy + 1e-8)
        ic_list.append(float(ic))
    return float(np.mean(ic_list)) if ic_list else 0.0


def backtest_one(formula, feat, raw_dict, target_ret, cost_rate=0.0001):
    """對一個公式跑完整回測，返回統計結果。"""
    vm = StackVM()
    factor = vm.execute(formula, feat)  # [N, T]
    if factor is None:
        return None

    N, T = factor.shape
    factor_np = factor.detach().numpy()
    target_np = target_ret.detach().numpy()

    # 持倉
    pos = compute_target_positions_stateless(factor)  # [N, T]
    pos_np = pos.detach().numpy()

    # 換手
    prev = np.zeros_like(pos_np)
    prev[:, 1:] = pos_np[:, :-1]
    turnover = np.abs(pos_np - prev)

    pnl = pos_np * target_np - turnover * cost_rate  # [N, T]

    # 分品種統計
    per_sym = {}
    sym_names = ["EURUSD", "USDJPY"][:N]
    for i, sym in enumerate(sym_names):
        p = pnl[i]
        cum = np.cumsum(p)
        per_sym[sym] = {
            "pnl":       p,
            "cum":       cum,
            "total_ret": float(cum[-1]),
            "sharpe":    calc_sharpe(p),
            "sortino":   calc_sortino(p),
            "mdd":       calc_mdd(cum),
            "n_trades":  int((np.abs(np.diff(pos_np[i])) > 0.5).sum()),
            "avg_hold":  float(T / max(1, int((np.abs(np.diff(pos_np[i])) > 0.5).sum()))),
        }

    # 組合
    port_pnl = pnl.mean(axis=0)
    port_cum = np.cumsum(port_pnl)

    # IC
    ic = calc_ic(factor_np, target_np)

    # 分段分析：前50% vs 後50%（檢查過擬合）
    split = T // 2
    p1 = port_pnl[:split]
    p2 = port_pnl[split:]

    return {
        "formula":        formula,
        "readable":       decode(formula),
        "per_sym":        per_sym,
        "port_total_ret": float(port_cum[-1]),
        "port_sharpe":    calc_sharpe(port_pnl),
        "port_sortino":   calc_sortino(port_pnl),
        "port_mdd":       calc_mdd(port_cum),
        "ic":             ic,
        "n_pos_syms":     sum(1 for d in per_sym.values() if d["total_ret"] > 0),
        "n_syms":         N,
        # 穩定性：前後半段
        "half1_sharpe":   calc_sharpe(p1),
        "half2_sharpe":   calc_sharpe(p2),
        "half1_sortino":  calc_sortino(p1),
        "half2_sortino":  calc_sortino(p2),
        # 換手
        "avg_turnover":   float(turnover.mean()),
        "avg_hold_h":     float(per_sym[sym_names[0]]["avg_hold"]) if sym_names else 0,
    }


def main():
    offline = "--offline" in sys.argv

    # ── 候選因子（從 elite pool 手工提取，按分數降序）
    CANDIDATES = [
        # score=6.636 — 當前最優
        [40, 51, 81, 83, 86, 125, 88, 125],
        # score=6.224 — 變體（TANH_SQUASH→TS_SUM_5）
        [40, 51, 81, 83, 86, 125, 88, 112],
        # score=5.960 — TRIX × HURST 類型
        [50, 105, 94, 103, 59, 66, 119, 88],
        # score=5.827 — TRIX × HURST（4 個等分變體取代表）
        [50, 105, 89, 103, 59, 88, 112, 66],
        # score=5.811
        [50, 105, 82, 103, 59, 88, 112, 66],
        # 冒煙測試最優（供參比）
        [3, 23, 110, 42, 53, 104, 72, 109],
    ]

    print(f"\n{'='*70}")
    print(f"  Elite Pool 批次回測  |  vocab={VOCAB_VERSION}  |  offline={offline}")
    print(f"{'='*70}\n")

    # ── 載入數據
    print("載入數據...")
    with MT5DataFetcher(offline=offline) as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        raw_dict   = mgr.raw_dict
        syms       = mgr.symbols
        T          = raw_dict["open"].shape[1]
        print(f"  品種: {syms}  T={T} bars")

        # 只取 forex 組（EURUSD/USDJPY）
        forex_syms = ["EURUSD", "USDJPY"]
        forex_idx  = [syms.index(s) for s in forex_syms if s in syms]
        forex_raw  = {k: v[forex_idx] for k, v in raw_dict.items()}
        forex_feat = MT5FeatureEngineer.compute_features(forex_raw)
        target_ret = mgr.target_ret[forex_idx]

        print(f"  Forex 品種: {[syms[i] for i in forex_idx]}  feat={forex_feat.shape}\n")

        # ── 批次回測
        results = []
        for i, formula in enumerate(CANDIDATES):
            res = backtest_one(formula, forex_feat, forex_raw, target_ret)
            if res is None:
                print(f"  [{i+1}] ❌ 公式執行失敗: {formula}")
                continue
            results.append(res)

    # ── 列印結果
    print(f"\n{'='*70}")
    print(f"  回測結果匯總（{T} bars H1，forex 組）")
    print(f"{'='*70}")
    hdr = f"  {'#':>3} {'Score*':>7} {'TotRet':>7} {'Sharpe':>7} {'Sortino':>8} {'MDD':>6} {'IC':>7} {'H1':>5} {'H2':>5} {'AvgHold':>8} {'公式類型':}"
    print(hdr)
    print(f"  {'─'*95}")

    for i, r in enumerate(results):
        # 判斷前後半段一致性
        h1, h2 = r["half1_sharpe"], r["half2_sharpe"]
        consistency = "✓" if h1 > 0 and h2 > 0 else ("⚠" if h1 * h2 > 0 else "✗")
        print(f"  [{i+1:2d}]  {r['port_sortino']:>7.3f}  "
              f"{r['port_total_ret']:>7.3f}  "
              f"{r['port_sharpe']:>7.3f}  "
              f"{r['port_sortino']:>8.3f}  "
              f"{r['port_mdd']:>6.3f}  "
              f"{r['ic']:>7.4f}  "
              f"{h1:>5.2f}  {h2:>5.2f}  "
              f"{r['avg_hold_h']:>8.1f}h  "
              f"{consistency} {r['readable'][:55]}")

    # ── 品種級詳情
    print(f"\n{'─'*70}")
    print(f"  品種級詳情（前 3 個公式）")
    print(f"{'─'*70}")
    for i, r in enumerate(results[:3]):
        print(f"\n  [{i+1}] {r['readable']}")
        for sym, d in r["per_sym"].items():
            sig = "✓" if d["total_ret"] > 0 else "✗"
            print(f"      {sym}: TotRet={d['total_ret']:+.3f}  Sharpe={d['sharpe']:+.3f}  "
                  f"Sortino={d['sortino']:+.3f}  MDD={d['mdd']:.3f}  "
                  f"Trades={d['n_trades']}  AvgHold={d['avg_hold']:.0f}h  {sig}")

    # ── 關鍵診斷
    print(f"\n{'='*70}")
    print(f"  關鍵診斷")
    print(f"{'='*70}")
    for i, r in enumerate(results):
        issues = []
        h1, h2 = r["half1_sharpe"], r["half2_sharpe"]
        if h1 > 0 and h2 < 0:
            issues.append(f"過擬合嫌疑: 前半Sharpe={h1:.2f} 後半={h2:.2f}")
        if r["port_mdd"] > 0.5:
            issues.append(f"最大回撤過大: {r['port_mdd']:.3f}")
        if r["avg_hold_h"] < 2:
            issues.append(f"持倉太短(>{r['avg_hold_h']:.1f}h)，點差侵蝕大")
        if r["avg_hold_h"] > 500:
            issues.append(f"持倉極長({r['avg_hold_h']:.0f}h)，實際交易數極少")
        if r["n_pos_syms"] < r["n_syms"]:
            issues.append(f"僅{r['n_pos_syms']}/{r['n_syms']}品種盈利，跨品種一致性差")
        if abs(r["ic"]) < 0.005:
            issues.append(f"IC≈0({r['ic']:.4f})，預測力存疑")
        if issues:
            print(f"  [{i+1}] {decode(r['formula'])[:50]}")
            for iss in issues:
                print(f"       ⚠ {iss}")
        else:
            print(f"  [{i+1}] ✓ 無明顯問題: {decode(r['formula'])[:55]}")

    print()
    # 保存
    report_path = "backtest_output/elite_backtest.json"
    Path("backtest_output").mkdir(exist_ok=True)
    summary = []
    for r in results:
        summary.append({
            "formula": r["formula"],
            "readable": r["readable"],
            "port_total_ret": round(r["port_total_ret"], 4),
            "port_sharpe":    round(r["port_sharpe"], 4),
            "port_sortino":   round(r["port_sortino"], 4),
            "port_mdd":       round(r["port_mdd"], 4),
            "ic":             round(r["ic"], 5),
            "half1_sharpe":   round(r["half1_sharpe"], 4),
            "half2_sharpe":   round(r["half2_sharpe"], 4),
            "avg_hold_h":     round(r["avg_hold_h"], 1),
        })
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"報告已保存 → {report_path}\n")


if __name__ == "__main__":
    main()
