"""
tests/unit/test_backtest.py — MT5Backtest 單元測試

驗證 COST_RATE 預設值、evaluate() 返回類型、
已知 PnL 序列的 Sortino 計算結果、換手率懲罰以及 80/20 分割點。

Requirements: 5.2, 5.3
"""
import math
import pytest
import torch

from model_core.backtest import MT5Backtest


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: COST_RATE 預設值
# ─────────────────────────────────────────────────────────────────────────────

def test_default_cost_rate():
    """MT5Backtest 默認 cost_rate 應為 0.0001（forex/metals 點差+佣金）。
    Requirements: 5.2
    """
    bt = MT5Backtest()
    assert bt.cost_rate == 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: evaluate() 返回正確類型和形狀
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_return_types():
    """evaluate() 應返回 (scalar Tensor, float)。
    Requirements: 5.2
    """
    bt = MT5Backtest()
    T = 100
    N = 2
    factors = torch.randn(N, T)
    target_ret = torch.randn(N, T) * 0.01

    score, oos = bt.evaluate(factors, {}, target_ret)

    assert isinstance(score, torch.Tensor), "score 應為 torch.Tensor"
    assert score.shape == torch.Size([]), "score 應為標量（零維張量）"
    assert isinstance(oos, float), "oos 應為 float"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: 簡單已知 PnL 序列 — 全部做多 + 固定正收益
# ─────────────────────────────────────────────────────────────────────────────

def test_simple_known_pnl():
    """全部做多且每期收益固定為 0.01 時，score 應為有限標量，不為 NaN/Inf。

    邏輯推導：
    - factors >> 0  →  tanh(factors) → +1  →  position = +1
    - target_ret = 0.01（每期）
    - 僅第 1 期有換手（從 0 → +1），後續換手為 0
    - pnl[0] = 1 * 0.01 - 1 * cost_rate ≈ 0.01 - 0.0001
    - pnl[t>0] = 1 * 0.01 - 0 = 0.01
    - 所有 pnl > 0，Sortino 分子 > 0，無下行收益 → 分母用 eps 代替
    - 期望 score 為大正數，且為有限值

    Requirements: 5.2, 5.3
    """
    bt = MT5Backtest()
    N, T = 1, 100

    # 極大正因子 → sign(tanh(10)) = +1
    factors = torch.full((N, T), 10.0)
    target_ret = torch.full((N, T), 0.01)

    score, oos = bt.evaluate(factors, {}, target_ret)

    assert torch.isfinite(score), f"score 應為有限值，實際為 {score.item()}"
    assert score.item() > 0, f"全部正收益時 score 應 > 0，實際為 {score.item()}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: 高換手率懲罰
# ─────────────────────────────────────────────────────────────────────────────

def test_high_turnover_penalty():
    """交替方向（高換手）的評分應嚴格低於恆定方向（低換手）的評分。

    構造：
    - high_turnover: factors 在 +10/-10 之間交替 → 每期換手 = 2
    - low_turnover:  factors 全為 +10           → 換手僅在首期

    兩者使用相同的正 target_ret，high_turnover 因懲罰 (score -= 1.0) 而得分更低。

    Requirements: 5.3
    """
    bt = MT5Backtest()
    N, T = 1, 100
    target_ret = torch.full((N, T), 0.005)

    # 交替因子 → position 在 +1/-1 間切換 → turnover.mean() >> 0.5
    alternating = torch.tensor(
        [10.0 if i % 2 == 0 else -10.0 for i in range(T)]
    ).unsqueeze(0)  # [1, T]

    # 恆定正因子 → position 恆為 +1 → turnover.mean() ≈ 0
    constant = torch.full((N, T), 10.0)

    score_high, _ = bt.evaluate(alternating, {}, target_ret)
    score_low, _ = bt.evaluate(constant, {}, target_ret)

    assert score_high.item() < score_low.item(), (
        f"高換手評分 ({score_high.item():.4f}) 應 < 低換手評分 ({score_low.item():.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: 80/20 分割點驗證
# ─────────────────────────────────────────────────────────────────────────────

def test_split_point_100():
    """T=100 時，in-sample 分割點應為 80。
    Requirements: 5.4 (通過檢查 OOS 收益來驗證分割點)
    """
    # 前 80 期收益為 +0.1，後 20 期收益為 -0.1
    # 若分割正確，score 基於前 80 期（正收益），oos 基於後 20 期（負收益）
    bt = MT5Backtest()
    T = 100
    N = 1

    factors = torch.full((N, T), 10.0)  # position = +1 everywhere

    target_ret = torch.zeros(N, T)
    target_ret[:, :80] = 0.1   # in-sample 高收益
    target_ret[:, 80:] = -0.1  # out-of-sample 負收益

    score, oos = bt.evaluate(factors, {}, target_ret)

    # in-sample 全正 → score > 0
    assert score.item() > 0, f"IS score 應 > 0（前80期正收益），實際為 {score.item()}"
    # oos 全負 → oos < 0
    assert oos < 0, f"OOS 均值應 < 0（後20期負收益），實際為 {oos}"


def test_split_point_50():
    """T=50 時，in-sample 分割點應為 40。
    Requirements: 5.4
    """
    bt = MT5Backtest()
    T = 50
    N = 1

    factors = torch.full((N, T), 10.0)  # position = +1 everywhere

    target_ret = torch.zeros(N, T)
    target_ret[:, :40] = 0.1   # in-sample 高收益（前 40 期）
    target_ret[:, 40:] = -0.1  # out-of-sample 負收益（後 10 期）

    score, oos = bt.evaluate(factors, {}, target_ret)

    assert score.item() > 0, f"IS score 應 > 0（前40期正收益），實際為 {score.item()}"
    assert oos < 0, f"OOS 均值應 < 0（後10期負收益），實際為 {oos}"


def test_split_point_exact_count():
    """直接驗證分割計數：in-sample = floor(T*0.8)，oos = T - split。
    Requirements: 5.4
    """
    for T in [10, 50, 100, 123, 200]:
        expected_split = math.floor(T * 0.8)
        expected_oos = T - expected_split

        bt = MT5Backtest()
        N = 1

        # 設計特殊 target_ret：前 split 期 = +1，後 oos 期 = -1
        factors = torch.full((N, T), 10.0)
        target_ret = torch.ones(N, T)
        target_ret[:, expected_split:] = -1.0

        score, oos = bt.evaluate(factors, {}, target_ret)

        # IS 全正 → score > 0；OOS 全負 → oos < 0
        assert score.item() > 0, (
            f"T={T}: IS score 應 > 0，期望分割={expected_split}，實際 {score.item()}"
        )
        assert oos < 0, (
            f"T={T}: OOS 均值應 < 0，期望 OOS={expected_oos} 期，實際 {oos}"
        )
