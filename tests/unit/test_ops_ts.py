"""
單元測試：時序運算元（TS_MEAN / TS_STD / TS_RANK / TS_CORR_10）及新增趨勢運算元

驗證：
- 輸出形狀均為 [N, T]
- TS_RANK_5/10/20 值域 ∈ [0, 1)
- TS_CORR_10 在常數輸入時輸出 0
- 所有運算元對邊界值（全零、極大值 1e8）無 NaN / Inf
- len(OPS_CONFIG) == 28（原 22 + 新增 6 個趨勢/動量運算元）

需求：F2.1~F2.6
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from model_core.ops import OPS_CONFIG, _ts_mean, _ts_std, _ts_rank, _ts_corr_10

# ── 常量 ────────────────────────────────────────────────────────────────────────
N, T = 4, 30

# 原始 10 個時序運算元（索引 12~21），不含新增的趨勢運算元
TS_OPS = OPS_CONFIG[12:22]


# ── 輔助：隨機正態輸入 ───────────────────────────────────────────────────────────
def rand_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(N, T)


# ── 1. OPS_CONFIG 長度驗證 ───────────────────────────────────────────────────────
class TestOpsConfigLength:
    def test_ops_config_length_equals_22(self):
        """OPS_CONFIG 共 28 個運算元（原 12 基礎 + 10 時序 + 6 趨勢/動量）"""
        assert len(OPS_CONFIG) == 28, (
            f"OPS_CONFIG 長度應為 28，實際為 {len(OPS_CONFIG)}"
        )

    def test_new_ops_count_equals_10(self):
        """時序運算元（索引 12-21）共 10 個"""
        assert len(TS_OPS) == 10

    def test_ts_op_names(self):
        """驗證 10 個新運算元名稱正確"""
        expected_names = [
            "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
            "TS_STD_5",  "TS_STD_10",  "TS_STD_20",
            "TS_RANK_5", "TS_RANK_10", "TS_RANK_20",
            "TS_CORR_10",
        ]
        actual_names = [name for name, _, _ in TS_OPS]
        assert actual_names == expected_names


# ── 2. 輸出形狀 [N, T] ──────────────────────────────────────────────────────────
class TestOutputShape:
    """全部 10 個新運算元：輸入 [N, T] → 輸出 [N, T]"""

    def _invoke(self, name, fn, arity):
        x = rand_input()
        if arity == 1:
            return fn(x)
        else:  # arity == 2（TS_CORR_10）
            y = rand_input() + 0.1  # 略加偏移，避免 x==y 完全相關
            return fn(x, y)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_output_shape(self, name, fn, arity):
        out = self._invoke(name, fn, arity)
        assert out.shape == (N, T), (
            f"{name}: 期望形狀 ({N}, {T})，實際 {tuple(out.shape)}"
        )


# ── 3. TS_RANK 值域 ∈ [0, 1) ────────────────────────────────────────────────────
class TestTsRankRange:
    """TS_RANK_5/10/20 輸出所有值滿足 0 ≤ v < 1"""

    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_rank_lower_bound(self, d):
        x = rand_input()
        out = _ts_rank(x, d)
        assert (out >= 0.0).all(), f"_ts_rank(x, {d}) 存在負值"

    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_rank_upper_bound_strict(self, d):
        x = rand_input()
        out = _ts_rank(x, d)
        assert (out < 1.0).all(), (
            f"_ts_rank(x, {d}) 存在 ≥ 1.0 的值（最大值={out.max().item():.6f}）"
        )

    @pytest.mark.parametrize("name,fn,arity", [
        (n, f, a) for n, f, a in TS_OPS if n.startswith("TS_RANK")
    ], ids=["TS_RANK_5", "TS_RANK_10", "TS_RANK_20"])
    def test_rank_range_via_ops_config(self, name, fn, arity):
        """通過 OPS_CONFIG 中的 lambda 驗證同一約束"""
        x = rand_input()
        out = fn(x)
        assert (out >= 0.0).all() and (out < 1.0).all(), (
            f"{name} 值域越界：min={out.min().item():.6f}, max={out.max().item():.6f}"
        )


# ── 4. TS_CORR_10 常數輸入輸出 0 ────────────────────────────────────────────────
class TestTsCorr10Constant:
    """當 x 或 y 在整個序列中為常數時，TS_CORR_10 在窗口完全填滿後（t >= 9）輸出應為 0。

    注意：_ts_corr_10 使用 d=10 的因果滑動窗口，左補零填充。
    在前 9 個時間步（t=0..8），窗口內混有零值和真實常數，std 不為零，
    因此 mask (std < 1e-6) 不生效，輸出可能非零。
    從第 10 步（t=9）起，窗口完全由真實常數填滿，std=0 觸發 mask，輸出為 0。
    """

    # d=10，滿窗口需要 10 個真實值，從 t=9 開始（0-indexed）
    _FULL_WINDOW_START = 9  # 即 T 維度索引 9 起（第 10 個時間步）

    def test_constant_x_outputs_zero_after_warmup(self):
        """x 為常數時，滿窗口之後（t >= 9）輸出應全為 0"""
        x = torch.ones(N, T) * 3.14
        y = rand_input()
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"x 為常數時 t>={self._FULL_WINDOW_START} 輸出應為 0，"
            f"實際最大絕對值={warmed.abs().max().item():.2e}"
        )

    def test_constant_y_outputs_zero_after_warmup(self):
        """y 為常數時，滿窗口之後（t >= 9）輸出應全為 0"""
        x = rand_input()
        y = torch.ones(N, T) * -2.71
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"y 為常數時 t>={self._FULL_WINDOW_START} 輸出應為 0，"
            f"實際最大絕對值={warmed.abs().max().item():.2e}"
        )

    def test_both_constant_outputs_zero_after_warmup(self):
        """x 和 y 均為常數時，滿窗口之後（t >= 9）輸出應全為 0"""
        x = torch.full((N, T), 1.0)
        y = torch.full((N, T), 2.0)
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6)

    def test_corr_10_via_ops_config_constant_after_warmup(self):
        """通過 OPS_CONFIG 中的 TS_CORR_10 條目驗證常數輸入的滿窗口行為"""
        _, fn, _ = next((name, fn, a) for name, fn, a in TS_OPS if name == "TS_CORR_10")
        x = torch.ones(N, T)
        y = rand_input()
        out = fn(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6)

    def test_long_constant_input_fully_zero(self):
        """使用更長序列（T=50）確保滿窗口大部分為零"""
        T_long = 50
        x = torch.full((N, T_long), 7.0)
        y = torch.randn(N, T_long)
        out = _ts_corr_10(x, y)
        # 從 t=9 起所有位置應為 0
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"長序列中 x 為常數時，滿窗口部分應全為 0，"
            f"實際最大絕對值={warmed.abs().max().item():.2e}"
        )


# ── 5. 無 NaN / Inf（邊界值測試）───────────────────────────────────────────────
class TestNoNanInf:
    """全部 10 個運算元對邊界輸入（全零、極大值 1e8）不產生 NaN / Inf"""

    @staticmethod
    def _check(name, out):
        assert not torch.isnan(out).any(), f"{name}: 輸出包含 NaN"
        assert not torch.isinf(out).any(), f"{name}: 輸出包含 Inf"

    def _invoke(self, fn, arity, x):
        if arity == 1:
            return fn(x)
        else:
            return fn(x, x.clone())  # TS_CORR_10：x==y → 常數窗口 → 輸出 0（已 mask）

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_zero_input(self, name, fn, arity):
        x = torch.zeros(N, T)
        out = self._invoke(fn, arity, x)
        self._check(name, out)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_large_input(self, name, fn, arity):
        x = torch.full((N, T), 1e8)
        out = self._invoke(fn, arity, x)
        self._check(name, out)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_random_input(self, name, fn, arity):
        """隨機正態輸入亦無 NaN / Inf"""
        x = rand_input()
        out = self._invoke(fn, arity, x)
        self._check(name, out)


# ── 6. 輔助函數單元測試 ─────────────────────────────────────────────────────────
class TestHelperFunctions:
    """直接測試 _ts_mean / _ts_std / _ts_rank / _ts_corr_10 的基本行為"""

    def test_ts_mean_shape(self):
        x = rand_input()
        assert _ts_mean(x, 5).shape == (N, T)
        assert _ts_mean(x, 10).shape == (N, T)
        assert _ts_mean(x, 20).shape == (N, T)

    def test_ts_std_non_negative(self):
        """滑動標準差加了 1e-6 下界，輸出應 ≥ 1e-6"""
        x = rand_input()
        for d in (5, 10, 20):
            out = _ts_std(x, d)
            assert (out >= 1e-7).all(), f"_ts_std(x, {d}) 存在 < 1e-7 的值"

    def test_ts_std_constant_near_eps(self):
        """全常數輸入的標準差：窗口填滿（t >= d-1）後應約等於 1e-6（僅來自下界偏移）。

        _ts_rolling 對長度 d 的窗口左補 d-1 個零，所以：
        - t=0..d-2：窗口包含零和真實常數，std > 0
        - t=d-1 起：窗口全為真實常數，std ≈ 0，加上 1e-6 下界後 ≈ 1e-6
        """
        x = torch.ones(N, T) * 5.0
        for d in (5, 10, 20):
            out = _ts_std(x, d)
            # 只檢查滿窗口部分（t >= d-1）
            warmed = out[:, d - 1:]
            assert torch.allclose(warmed, torch.full_like(warmed, 1e-6), atol=1e-7), (
                f"_ts_std 常數輸入（滿窗口 t>={d-1}）應接近 1e-6，實際最大={warmed.max().item():.2e}"
            )

    def test_ts_corr_10_range(self):
        """正常隨機輸入時，相關係數值域應在 [-1, 1]"""
        torch.manual_seed(0)
        x = torch.randn(N, T)
        y = torch.randn(N, T)
        out = _ts_corr_10(x, y)
        assert (out >= -1.0 - 1e-5).all() and (out <= 1.0 + 1e-5).all(), (
            f"TS_CORR_10 值域越界：min={out.min().item():.6f}, max={out.max().item():.6f}"
        )
