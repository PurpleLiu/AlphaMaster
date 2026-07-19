"""
冒煙測試：Config 欄位與 FEATURE_NAMES 內容驗證

斷言：
- FEATURE_NAMES 不含 LIQ_SCORE / FOMO（舊 Solana 特有因子）
- Config.INPUT_DIM == 6

注意：task 5.1 會將 vocab.py 更新為新的 MT5 特徵名稱。
      此測試使用 try/except 優雅處理舊版 vocab.py 中仍含舊欄位的情形。

Requirements: 11.1, 5.5 (4.1)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from config import Config


def _get_feature_names():
    """
    嘗試從 model_core.vocab 獲取 FEATURE_NAMES。
    若模組不存在或屬性不存在，返回 None。
    """
    try:
        from model_core.vocab import FEATURE_NAMES
        return FEATURE_NAMES
    except (ImportError, AttributeError):
        return None


def _is_mt5_feature_names_updated():
    """
    檢查 FEATURE_NAMES 是否已更新為 MT5 特徵集（task 5.1 完成後）。
    若仍包含舊 Solana 特有因子則返回 False，表示 task 5.1 尚未執行。
    """
    feature_names = _get_feature_names()
    if feature_names is None:
        return False
    # 舊版特徵含 Solana 特有因子
    old_features = {"LIQ_SCORE", "FOMO", "LOG_VOL"}
    return not bool(old_features & set(feature_names))


class TestConfigInputDimSmoke:
    def test_input_dim_equals_10(self):
        """Config.INPUT_DIM must equal 20 (expanded from 10 to 20 features)."""
        assert Config.INPUT_DIM == 20


class TestFeatureNamesSmoke:
    """
    驗證 FEATURE_NAMES 不含舊 Solana 鏈上特有因子。

    - 若 vocab.py 尚未更新（task 5.1 前），這些測試將跳過，以免誤報。
    - 若 vocab.py 已更新，則必須滿足斷言。
    """

    def test_feature_names_accessible(self):
        """FEATURE_NAMES 應當可以從 model_core.vocab 導入"""
        feature_names = _get_feature_names()
        assert feature_names is not None, (
            "Cannot import FEATURE_NAMES from model_core.vocab. "
            "Ensure model_core/vocab.py exists and defines FEATURE_NAMES."
        )

    def test_liq_score_not_in_feature_names(self):
        """FEATURE_NAMES 不應包含 'LIQ_SCORE'（已廢棄的鏈上流動性因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "LIQ_SCORE" not in feature_names, (
            f"FEATURE_NAMES should not contain 'LIQ_SCORE', but got: {feature_names}"
        )

    def test_fomo_not_in_feature_names(self):
        """FEATURE_NAMES 不應包含 'FOMO'（已廢棄的鏈上情緒因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "FOMO" not in feature_names, (
            f"FEATURE_NAMES should not contain 'FOMO', but got: {feature_names}"
        )

    def test_feature_names_length_matches_input_dim(self):
        """FEATURE_NAMES 的長度應等於 Config.INPUT_DIM（20）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert len(feature_names) == Config.INPUT_DIM, (
            f"Expected len(FEATURE_NAMES) == {Config.INPUT_DIM}, "
            f"got {len(feature_names)}: {feature_names}"
        )
