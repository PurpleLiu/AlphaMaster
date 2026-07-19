"""
model_core/backtest.py — MT5 回測評估器（組合級多目標 Reward）

評分框架（5品種組合版）：
  final_score =
      0.35 * portfolio_sortino          # 組合整體風險調整收益
    + 0.20 * portfolio_calmar           # 組合整體回撤控制
    + 0.15 * ts_ic_stability            # 時序IC穩定性（比橫截面IC更重要）
    + 0.10 * symbol_consistency         # 品種一致性（防止單品種拖累）
    + 0.10 * cost_stress                # 成本壓力測試（2x成本下仍盈利）
    + 0.10 * turnover_quality           # 換手率質量（交易頻率獎勵）
    - complexity_penalty                # 公式長度懲罰
    - correlation_penalty               # 因子相關性懲罰（由 engine 施加）

symbol_consistency 規則：
  - N 個品種中至少 ceil(N*0.6) 個 Sortino > 0 → 正分
  - 任何品種 Sortino < -2.0 → 重懲罰
  - 全部品種 Sortino > 0 → 額外獎勵
"""
import math
import torch
from torch import Tensor

from strategy_manager.signal import compute_target_positions_stateless
from .config import ModelConfig

_H1_PERIODS_PER_YEAR = 6240
_SORTINO_CLIP        = 20.0

_SECONDS_PER_YEAR = 365.25 * 86400.0


def estimate_periods_per_year(times) -> int:
    """從時間戳序列估計「每年 bar 數」（年化因子）。

    用 T / years_span，years_span = (t_last - t_first) / SECONDS_PER_YEAR。
    該方法自動適應不同市場（A 股 / 外匯 / 加密）與週期——因為數據裡只包含
    交易時段的 bar，跨日歷年的 bar 密度天然反映了該市場的交易頻率，
    無需按週期/市場寫死。

    替代了原先全局寫死 _H1_PERIODS_PER_YEAR=6240（僅外匯 H1 正確）的做法：
    A 股日線（~244 bar/年）、A 股 15min（~3904 bar/年）、加密日線（365 bar/年）
    都會被正確年化，不再被按 H1 放大/縮小。

    Args:
        times: [N, T] 或 [T] 的 Unix 秒時間戳（torch.Tensor 或 np.ndarray）。

    Returns:
        int，每年 bar 數；數據不足時回退到 _H1_PERIODS_PER_YEAR。
    """
    import numpy as _np
    if hasattr(times, "detach"):
        arr = times.detach().cpu().numpy()
    else:
        arr = _np.asarray(times)
    arr = arr.astype(_np.float64)
    if arr.ndim == 2:
        t_count = arr.shape[1]
        row = arr[0]
    elif arr.ndim == 1:
        t_count = arr.shape[0]
        row = arr
    else:
        return _H1_PERIODS_PER_YEAR
    if t_count < 2:
        return _H1_PERIODS_PER_YEAR
    span = float(row[-1] - row[0])
    if span <= 0:
        return _H1_PERIODS_PER_YEAR
    years = span / _SECONDS_PER_YEAR
    if years <= 0:
        return _H1_PERIODS_PER_YEAR
    ppy = t_count / years
    # 合理範圍鉗制，防止異常時間戳產生極端值
    return int(max(10, min(600000, round(ppy))))


class MT5Backtest:
    """MT5 組合級回測評估器。"""

    def __init__(
        self,
        cost_rate:        float = 0.0001,
        periods_per_year: int   = _H1_PERIODS_PER_YEAR,
    ):
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year

    # ──────────────────────────────────────────────────────────────────────
    # 基礎統計
    # ──────────────────────────────────────────────────────────────────────

    def _sortino(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        flat     = pnl.reshape(-1)
        mean_pnl = flat.mean()
        downside = flat[flat < 0]
        raw_std  = downside.std(unbiased=False) if downside.numel() > 0 \
                   else torch.tensor(0.0, dtype=flat.dtype, device=flat.device)
        # P0b 修復：下行標準差地板改為全序列 std 的 20%，防止稀疏 PnL 靠極小分母刷高分。
        # 原來 floor=|mean_pnl| 對稀疏序列趨近於零，導致 Sortino 爆炸。
        full_std       = flat.std(unbiased=False).clamp(min=eps)
        floor          = torch.clamp(full_std * 0.2, min=eps)
        downside_std   = torch.clamp(raw_std, min=floor)
        sortino        = mean_pnl / downside_std * math.sqrt(self.periods_per_year)
        return torch.clamp(sortino, -_SORTINO_CLIP, _SORTINO_CLIP)

    def _calmar(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        """Calmar = annualized_return / max_drawdown（截斷到 [-10, 10]）。"""
        flat      = pnl.reshape(-1)
        ann_ret   = flat.mean() * self.periods_per_year
        cum       = torch.cumsum(flat, dim=0)
        peak      = torch.cummax(cum, dim=0).values
        drawdown  = (peak - cum).max()
        drawdown  = torch.clamp(drawdown, min=eps)
        calmar    = ann_ret / drawdown
        return torch.clamp(calmar, -10.0, 10.0)

    # ──────────────────────────────────────────────────────────────────────
    # 組合級評分組件
    # ──────────────────────────────────────────────────────────────────────

    def _ts_ic_stability(self, factors: Tensor, target_ret: Tensor) -> float:
        """時序 IC 穩定性：每個品種內部 factor[t] 與 ret[t+1] 的相關性均值。

        比橫截面 IC 更適合 5 品種宇宙（橫截面 N=5 統計意義弱）。

        Returns:
            float，約 [-1, 1]，正值代表因子有預測力。
        """
        N, T = factors.shape
        if T < 10:
            return 0.0

        ic_list = []
        for n in range(N):
            x = factors[n, :-1]
            y = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic.item())

        if not ic_list:
            return 0.0

        ic_mean = sum(ic_list) / len(ic_list)
        ic_std  = (sum((v - ic_mean) ** 2 for v in ic_list) / len(ic_list)) ** 0.5
        # 穩定性 = IC均值 / IC標準差（IR，截斷到 [-3, 3]）
        stability = ic_mean / (ic_std + 1e-6)
        return float(max(-3.0, min(3.0, stability)))

    def _symbol_consistency(
        self,
        per_symbol_sortino: list[float],
        per_symbol_trade_count: list[int] | None = None,
        eval_bars: int = 0,
    ) -> float:
        """品種一致性懲罰/獎勵。

        規則（優先度從高到低）：
        1. 無交易品種超過 40%：重懲罰 -3.0
        2. P0a 新增：有交易的品種中，交易筆數 < eval_bars/100 (約每100bar少於1筆)
           視為"稀疏有效"，等同無效。防止3~6筆偶發交易刷高 Sortino。
        3. 任何品種 Sortino < -2.0：重懲罰 -2.0
        4. 有效品種中正收益比例決定獎懲
        """
        N = len(per_symbol_sortino)
        if N == 0:
            return 0.0

        # 最小有效交易數：每 100 bar 至少 1 筆，下限 5 筆
        min_trades = max(5, eval_bars // 100) if eval_bars > 0 else 5

        # 重新判定"活躍"品種（必須交易數 >= min_trades）
        if per_symbol_trade_count is not None:
            n_inactive = sum(1 for c in per_symbol_trade_count if c < min_trades)
            inactive_ratio = n_inactive / N
            if inactive_ratio > 0.4:
                return -3.0
        else:
            n_inactive = 0
            inactive_ratio = 0.0

        if any(s < -2.0 for s in per_symbol_sortino):
            return -2.0

        if per_symbol_trade_count is not None:
            active_sortinos = [
                s for s, c in zip(per_symbol_sortino, per_symbol_trade_count)
                if c >= min_trades
            ]
        else:
            active_sortinos = per_symbol_sortino

        if not active_sortinos:
            return -3.0

        n_positive = sum(1 for s in active_sortinos if s > 0)
        ratio = n_positive / len(active_sortinos)

        if ratio < 0.6:
            score = (ratio - 0.6) / 0.6 * 1.0
        else:
            score = (ratio - 0.6) / 0.4 * 1.0

        if ratio == 1.0:
            score += 0.5

        return float(score)

    def _cost_stress(
        self,
        position:   Tensor,
        target_ret: Tensor,
        stress_mult: float = 2.0,
    ) -> float:
        """成本壓力測試：2 倍成本下的 Sortino 是否還 > 0。

        Returns:
            float，壓力測試 Sortino（截斷到 [-5, 5]）。
        """
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        stressed_pnl = position * target_ret - turnover * self.cost_rate * stress_mult
        sortino = self._sortino(stressed_pnl)
        return float(torch.clamp(sortino, -5.0, 5.0))

    def _turnover_quality(self, position: Tensor) -> float:
        """交易頻率質量獎勵（每天約 1 筆為最優）。

        目標：每 12 bar 一筆（H1 每天約一筆）。
        """
        N, T = position.shape
        pos_2d = position.tolist()
        all_runs, total_trades = [], 0

        for n in range(N):
            runs, cur_len, cur_dir = [], 0, 0
            for p in pos_2d[n]:
                pi = int(p)
                if pi != 0:
                    if pi == cur_dir:
                        cur_len += 1
                    else:
                        if cur_len > 0: runs.append(cur_len)
                        cur_dir, cur_len = pi, 1
                else:
                    if cur_len > 0: runs.append(cur_len)
                    cur_dir, cur_len = 0, 0
            if cur_len > 0: runs.append(cur_len)
            all_runs.extend(runs)
            total_trades += len(runs)

        total_bars    = N * T
        target_trades = total_bars / 12.0
        actual_ratio  = total_trades / max(target_trades, 1.0)

        if actual_ratio <= 0:
            freq_score = -2.0
        elif actual_ratio < 0.05:
            freq_score = -2.0 + actual_ratio / 0.05
        elif actual_ratio < 0.5:
            freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
        elif actual_ratio <= 2.0:
            log_r = math.log(actual_ratio) / math.log(2.0)
            freq_score = 1.0 * math.exp(-0.5 * log_r ** 2)
        elif actual_ratio <= 8.0:
            freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
        else:
            freq_score = -2.0

        hold_bonus = 0.0
        if all_runs:
            avg_hold = sum(all_runs) / len(all_runs)
            hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)

        return float(freq_score + hold_bonus)

    def _beta_neutral_penalty(self, position: Tensor) -> float:
        """Beta 中性懲罰：多空比例嚴重失衡時扣分。

        因子輸出 >85% 同方向時，說明不是 alpha 因子而是 beta 因子
        （如 index 組的 TS_RANK 連續使用導致恆正輸出）。

        Returns:
            float，懲罰值（負數或零）
        """
        flat = position.reshape(-1)
        long_ratio = (flat > 0.05).float().mean().item()
        short_ratio = (flat < -0.05).float().mean().item()
        max_ratio = max(long_ratio, short_ratio)
        if max_ratio > 0.85:
            # 超過 85% 同方向，重罰
            excess = (max_ratio - 0.85) / 0.15  # 0~1
            return -2.0 * excess  # 最多 -2.0
        elif max_ratio > 0.70:
            # 70-85% 輕度失衡，輕罰
            excess = (max_ratio - 0.70) / 0.15  # 0~1
            return -0.5 * excess  # 最多 -0.5
        return 0.0

    def _half_consistency_bonus(self, pnl: Tensor) -> float:
        """前後一致性獎勵：前半段和後半段 Sortino 同號時加分。

        防止因子只在某一段市場環境（如牛市）有效。

        Returns:
            float，獎勵/懲罰值
        """
        T = pnl.shape[1]
        if T < 20:
            return 0.0
        half = T // 2
        s1 = self._sortino(pnl[:, :half]).item()
        s2 = self._sortino(pnl[:, half:]).item()
        if s1 > 0 and s2 > 0:
            return 0.5  # 前後都賺錢，獎勵
        elif s1 * s2 < 0:
            return -1.0  # 前後相反，重罰（如 index 組的 beta 因子）
        return 0.0  # 一正一零或兩零，不獎不罰

    def _exposure_penalty(self, position: Tensor) -> float:
        """在場時間懲罰（僅下限，無上限）：收益優先模式。

        只懲罰極稀疏交易（<10%在場），不懲罰高在場時間。
        高在場時間（滿倉趨勢跟蹤）是外匯市場最賺錢的形態之一，不應受罰。
        """
        flat = position.reshape(-1).abs()
        exposure = flat.mean().item()   # 連續倉位：均值即平均持倉量
        if exposure < 0.10:
            # 極稀疏：平均持倉 < 10% → 線性懲罰 [-2, 0)
            return float((exposure / 0.10 - 1.0) * 2.0)
        return 0.0

    def _turnover_penalty(self, turnover: Tensor) -> Tensor:
        """梯度式換手率懲罰。"""
        mean_to = turnover.mean()
        penalty = torch.clamp(
            (mean_to - 0.2) * 3.0,
            min=0.0,
            max=3.0,
        )
        return -penalty

    # ──────────────────────────────────────────────────────────────────────
    # Walk-Forward 輔助介面
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_fold(
        self,
        factors:     Tensor,
        target_ret:  Tensor,
        train_start: int,
        train_end:   int,
        val_start:   int,
        val_end:     int,
    ) -> tuple[Tensor, Tensor]:
        """在指定訓練/驗證切片上計算組合多目標得分。

        train_score：用於 REINFORCE 梯度更新（in-sample 多目標）。
        val_score：用於選冠軍，加入 OOS Sortino 門控：
          - OOS Sortino <= 0：乘以 0.1~0.5 懲罰，強制冠軍必須在驗證段盈利
          - OOS Sortino > 0：乘以最多 1.2 獎勵
        """
        position = compute_target_positions_stateless(factors)  # neutral band

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * self.cost_rate

        pnl_train = pnl[:, train_start:train_end]
        pnl_val   = pnl[:, val_start:val_end]

        # 訓練段：多目標 + 換手率懲罰
        train_bars = train_end - train_start
        train_score = self._multi_objective(
            factors[:, train_start:train_end],
            target_ret[:, train_start:train_end],
            pnl_train,
            position[:, train_start:train_end],
            eval_bars=train_bars,
        ) + self._turnover_penalty(turnover[:, train_start:train_end])

        # 驗證段：多目標 × OOS Sortino 門控
        val_bars = val_end - val_start
        base_val    = self._multi_objective(
            factors[:, val_start:val_end],
            target_ret[:, val_start:val_end],
            pnl_val,
            position[:, val_start:val_end],
            eval_bars=val_bars,
        )
        oos_sor = self._sortino(pnl_val).item()
        if oos_sor <= 0:
            # OOS虧損：重懲罰（Sortino=-1 → mult=0.1；Sortino=0 → mult=0.5）
            mult = max(0.1, 0.5 + oos_sor * 0.4)
        else:
            # OOS盈利：輕獎勵（最多+20%）
            mult = min(1.2, 1.0 + oos_sor * 0.1)
        val_score = base_val * mult

        return train_score, val_score

    def _reversal_bonus(self, factors: Tensor) -> Tensor:
        """反轉獎勵：鼓勵因子有低/負自相關（均值回歸特徵）。

        計算每個品種的 lag-1 自相關係數，越接近 0 或負值 = 越好。
        高度正自相關（>0.5）= 趨勢跟蹤，減分。

        單品種模式：直接返回標量。
        """
        N = factors.shape[0]
        scores = []
        for n in range(N):
            x = factors[n, :-1]     # t=0..T-2
            y = factors[n, 1:]      # t=1..T-1
            xm = x - x.mean(); ym = y - y.mean()
            sx = (xm**2).mean().sqrt(); sy = (ym**2).mean().sqrt()
            ac1 = (xm*ym).mean() / (sx*sy + 1e-8) if sx > 1e-6 and sy > 1e-6 else torch.tensor(0.0)
            # 獎勵低自相關：bonus = 1 - |ac1|, 負自相關額外加分
            bonus = 1.0 - torch.abs(ac1)
            if ac1 < 0:
                bonus = bonus + 0.5  # 負自相關（真正反轉）額外加分
            bonus = torch.clamp(bonus, -1.0, 2.0)
            scores.append(bonus)
        return torch.stack(scores).mean()

    def _symmetry_check(self, position: Tensor) -> Tensor:
        """多空對稱性檢查：獎勵 50/50 多空分布。

        均值回歸策略應該在多空之間大致平衡，
        過度偏向某一側 = 趨勢跟蹤特徵，應懲罰。
        """
        long_ratio  = (position > 0).float().mean()
        short_ratio = (position < 0).float().mean()
        # 理想值：long_ratio ≈ 0.5, short_ratio ≈ 0.5
        # 偏差：|long_ratio - 0.5| + |short_ratio - 0.5|
        deviation = torch.abs(long_ratio - 0.5) + torch.abs(short_ratio - 0.5)
        # 偏差 0 → 獎勵 1.0; 偏差 1.0 → 獎勵 -1.0
        bonus = 1.0 - 2.0 * deviation
        return torch.clamp(bonus, -1.0, 1.0)

    def _multi_objective(
        self,
        factors:    Tensor,
        target_ret: Tensor,
        pnl:        Tensor,
        position:   Tensor,
        eval_bars:  int = 0,
    ) -> Tensor:
        """收益優先的多目標評分（2026-07-04 重構）。

        核心改變：加入年化絕對收益項（權重 0.40），這是最主要的最佳化目標。
        Sortino/Calmar 權重大幅下調，僅作為風險調整輔助。
        clamp 上限放開（Sortino 40→20 保持，收益無上限）。

        N=1 單品種模式權重略有不同（無 symbol_consistency/cost_stress）。

        2026-07-08: 新增 forex 模式 — 偏向均值回歸策略。
        """
        N = pnl.shape[0]

        # ── 絕對收益（年化 log return）──────────────────────────────────
        # 連續倉位 pnl = position * target_ret - turnover * cost。
        # pnl.mean() 已是單 bar 平均收益，因此年化只乘每年 bar 數；不能再除以樣本長度。
        ann_ret = pnl.mean() * self.periods_per_year   # 標量張量，無截斷

        port_sortino = self._sortino(pnl)
        port_calmar  = self._calmar(pnl)
        ts_ic        = self._ts_ic_stability(factors, target_ret)
        tq           = self._turnover_quality(position)
        exp_pen      = self._exposure_penalty(position)

        if N == 1:
            beta_pen = self._beta_neutral_penalty(position)
            consist = self._half_consistency_bonus(pnl)

            if ModelConfig.REWARD_MODE == "forex":
                # Forex 均值回歸模式：
                #   - 降年化收益權重 (0.80→0.25)：外匯趨勢弱，避免獎勵虛假趨勢
                #   - 提 IC 權重 (0.03→0.25)：信號質量是核心
                #   - 新增反轉獎勵 (0.20)：獎勵低/負因子自相關
                #   - 新增對稱檢查 (0.15)：獎勵 50/50 多空平衡
                rev_bonus = self._reversal_bonus(factors)
                sym_bonus = self._symmetry_check(position)
                return (
                    0.25 * ann_ret           # 年化收益（降權，外匯趨勢噪聲大）
                    + 0.05 * port_sortino    # 風險調整輔助
                    + 0.05 * port_calmar     # 回撤控制
                    + 0.25 * ts_ic           # 信號質量（大幅提權）
                    + 0.20 * rev_bonus       # 反轉獎勵（核心：反趨勢）
                    + 0.15 * sym_bonus       # 多空對稱（均值回歸特徵）
                    + 0.05 * tq              # 交易頻率質量
                    + exp_pen                # 稀疏懲罰
                    + beta_pen               # Beta 中性懲罰
                    + consist                # 前後一致性獎懲
                )

            if ModelConfig.REWARD_MODE == "ftmo":
                # FTMO 專屬：年化收益 0.80，Calmar 0.10（控制 MDD 貼近 10% 上限）
                return (
                    0.80 * ann_ret           # 主目標：年化絕對收益（FTMO 加權）
                    + 0.05 * port_sortino    # 風險調整輔助（降權）
                    + 0.10 * port_calmar     # 回撤控制（保持，對齊 10% Max Loss）
                    + 0.03 * ts_ic           # IC 預測方向（降權）
                    + 0.02 * tq              # 交易頻率質量（降權）
                    + exp_pen                # 稀疏懲罰
                    + beta_pen               # Beta 中性懲罰
                    + consist                # 前後一致性獎懲
                )
            return (
                0.60 * ann_ret           # 主目標：年化絕對收益
                + 0.15 * port_sortino    # 風險調整輔助
                + 0.10 * port_calmar     # 回撤控制輔助
                + 0.10 * ts_ic           # IC 預測方向
                + 0.05 * tq              # 交易頻率質量
                + exp_pen                # 稀疏懲罰
                + beta_pen               # Beta 中性懲罰
                + consist                # 前後一致性獎懲
            )

        per_sym_sortino     = []
        per_sym_trade_count = []
        for n in range(N):
            per_sym_sortino.append(self._sortino(pnl[n]).item())
            # 連續倉位下，用 |position| 變化來估算交易次數
            pos_n = position[n].abs()
            # 視 tanh 輸出均值作為持倉量，換手次數用前後差異估計
            diff = (pos_n[1:] - pos_n[:-1]).abs()
            trades = int((diff > 0.1).sum().item())
            per_sym_trade_count.append(trades)

        sym_cons = self._symbol_consistency(
            per_sym_sortino, per_sym_trade_count, eval_bars=eval_bars
        )
        cost_s   = self._cost_stress(position, target_ret)
        beta_pen = self._beta_neutral_penalty(position)
        consist  = self._half_consistency_bonus(pnl)

        if ModelConfig.REWARD_MODE == "ftmo":
            # FTMO 專屬：年化收益 0.75（提權），Calmar 0.10（對齊 10% Max Loss）
            return (
                0.75 * ann_ret               # 主目標：年化絕對收益（FTMO 加權）
                + 0.05 * port_sortino        # 風險調整輔助（降權）
                + 0.10 * port_calmar         # 回撤控制（提權，控制 MDD）
                + 0.02 * ts_ic               # IC 預測方向（降權）
                + 0.03 * sym_cons            # 品種一致性（降權）
                + 0.02 * cost_s              # 成本壓力測試（降權）
                + 0.03 * tq                  # 交易頻率質量（降權）
                + exp_pen                    # 稀疏懲罰
                + beta_pen                   # Beta 中性懲罰
                + consist                    # 前後一致性獎懲
            )

        return (
            0.60 * ann_ret               # 主目標：年化絕對收益
            + 0.10 * port_sortino        # 風險調整輔助
            + 0.05 * port_calmar         # 回撤控制輔助
            + 0.10 * ts_ic               # IC 預測方向
            + 0.05 * sym_cons            # 品種一致性
            + 0.05 * cost_s              # 成本壓力測試
            + 0.05 * tq                  # 交易頻率質量
            + exp_pen                    # 稀疏懲罰
            + beta_pen                   # Beta 中性懲罰
            + consist                    # 前後一致性獎懲
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公開介面（非 Walk-Forward 模式）
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        factors:    Tensor,
        raw_dict:   dict,
        target_ret: Tensor,
    ) -> tuple[Tensor, float]:
        """評估一組 Alpha 因子（含 OOS 80/20 門控）。"""
        position = compute_target_positions_stateless(factors)

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * self.cost_rate

        T     = factors.shape[1]
        split = int(math.floor(T * 0.8))

        score = self._multi_objective(
            factors[:, :split], target_ret[:, :split],
            pnl[:, :split], position[:, :split],
            eval_bars=split,
        ) + self._turnover_penalty(turnover[:, :split])

        # OOS 門控（最後 20%）
        pnl_oos = pnl[:, split:]
        oos_sor = self._sortino(pnl_oos).item()
        if oos_sor <= 0:
            mult = max(0.1, 0.5 + oos_sor * 0.4)
            score = score * mult
        else:
            score = score * min(1.2, 1.0 + oos_sor * 0.1)

        mean_oos = pnl_oos.mean().item()
        return score, mean_oos
