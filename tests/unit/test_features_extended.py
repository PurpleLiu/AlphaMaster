"""
單元測試：MT5FeatureEngineer 擴展特徵驗證

覆蓋 compute_features 的形狀、數值安全性、各特徵的值域與前綴約束。

需求：F1.1~F1.7, F4.1~F4.4
"""
import pytest
import torch
import sys
import os

# 確保項目根目錄在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_core.features import MT5FeatureEngineer


# ─── 測試用 OHLCV fixture ─────────────────────────────────────────────────────

def _make_raw_dict(N: int = 3, T: int = 50, seed: int = 42) -> dict:
    """
    生成合法的隨機 OHLCV 字典。
    - close, open : rand(N, T) + 1.0  → 正值，範圍 [1, 2)
    - high        : max(close, open) + rand * 0.5  → high >= close, open
    - low         : min(close, open) - rand * 0.5  → low  <= close, open（但 > 0）
    - volume      : rand(N, T) * 100 + 1.0  → 正值
    """
    torch.manual_seed(seed)
    close  = torch.rand(N, T) + 1.0
    open_  = torch.rand(N, T) + 1.0
    noise  = torch.rand(N, T) * 0.5
    high   = torch.maximum(close, open_) + noise
    low    = (torch.minimum(close, open_) - noise).clamp(min=1e-3)
    volume = torch.rand(N, T) * 100.0 + 1.0
    return {
        "close":  close,
        "open":   open_,
        "high":   high,
        "low":    low,
        "volume": volume,
    }


# ─── 1. 輸出形狀 ──────────────────────────────────────────────────────────────

class TestComputeFeaturesShape:
    """compute_features 輸出形狀應為 [N, 20, T]（擴展自10，需求 F1.1, F1.2）"""

    def test_output_shape_default(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.shape == (3, 20, 50), (
            f"Expected shape (3, 20, 50), got {tuple(out.shape)}"
        )

    def test_output_ndim(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.ndim == 3

    def test_feature_dim_equals_10(self):
        """feature 維度固定為 20，對應 INPUT_DIM（需求 F1.7）"""
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.shape[1] == 20

    def test_time_dim_preserved(self):
        """T 維度應與輸入完全一致（需求 F1.1）"""
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.shape[2] == 50


# ─── 2. 數值安全：無 NaN / Inf ────────────────────────────────────────────────

class TestNoNanInf:
    """輸出中不應含有 NaN 或 Inf（需求 F4.3~F4.6）"""

    def test_no_nan(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_no_inf(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert not torch.isinf(out).any(), "Output contains Inf values"

    def test_no_nan_inf_combined(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert torch.isfinite(out).all(), "Output contains non-finite values (NaN or Inf)"


# ─── 3. PRESSURE（索引 3）值域 ∈ [-1, 1] ─────────────────────────────────────

class TestPressureRange:
    """PRESSURE（索引 12）所有值應被 clamp 至 [-1, 1]（需求 F4.1）"""

    def test_pressure_leq_1(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        pressure = out[:, 12, :]   # PRESSURE is now index 12 in 20-feature vocab
        assert (pressure <= 1.0).all(), (
            f"PRESSURE has values > 1.0, max={pressure.max().item():.4f}"
        )

    def test_pressure_geq_neg1(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        pressure = out[:, 12, :]
        assert (pressure >= -1.0).all(), (
            f"PRESSURE has values < -1.0, min={pressure.min().item():.4f}"
        )

    def test_pressure_range_strict(self):
        """一次性驗證 [-1, 1] 雙側邊界（需求 F4.1）"""
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        pressure = out[:, 12, :]
        assert pressure.abs().max().item() <= 1.0 + 1e-6, (
            "PRESSURE violates [-1, 1] bound"
        )


# ─── 4. ATR 原始值非負（在 log1p 前）────────────────────────────────────────

class TestAtrRawNonNegative:
    """_atr 原始輸出（log1p 壓縮前）應全部非負（需求 F1.3）"""

    def test_atr_raw_nonnegative(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        atr_raw = MT5FeatureEngineer._atr(close, high, low)
        assert (atr_raw >= 0).all(), (
            f"ATR raw has negative values, min={atr_raw.min().item():.6f}"
        )

    def test_atr_raw_shape(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        atr_raw = MT5FeatureEngineer._atr(close, high, low)
        assert atr_raw.shape == (3, 50)

    def test_atr_raw_no_nan(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        atr_raw = MT5FeatureEngineer._atr(close, high, low)
        assert not torch.isnan(atr_raw).any(), "ATR raw contains NaN"


# ─── 5. RET20（索引 8）前 20 個位置應為 0 ────────────────────────────────────

class TestRet20PrefixZero:
    """RET20 原始輸出的前 20 個時間步應等於 0（需求 F1.5）"""

    def test_ret20_raw_first20_are_zero(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        ret20_raw = MT5FeatureEngineer._ret20(close)
        prefix = ret20_raw[:, :20]
        assert (prefix == 0.0).all(), (
            f"RET20 raw first 20 positions are not all zero; "
            f"max abs = {prefix.abs().max().item():.6f}"
        )

    def test_ret20_raw_shape(self):
        raw = _make_raw_dict(N=3, T=50)
        close = raw["close"].float()
        ret20_raw = MT5FeatureEngineer._ret20(close)
        assert ret20_raw.shape == (3, 50)

    def test_ret20_after_position20_nonzero(self):
        """位置 20 之後至少部分值應非零（驗證計算邏輯未全零化）"""
        raw = _make_raw_dict(N=3, T=50)
        close = raw["close"].float()
        ret20_raw = MT5FeatureEngineer._ret20(close)
        suffix = ret20_raw[:, 20:]
        assert suffix.abs().max().item() > 0.0, (
            "RET20 suffix (positions 20+) is unexpectedly all zero"
        )
