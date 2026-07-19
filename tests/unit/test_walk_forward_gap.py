"""
單元測試：_build_walk_forward_folds 的 gap 機制

驗證以下需求：
  T2.1 – 正常情況下每折 val_start == train_end + gap
  T2.2 – 越界折不被添加（摺疊不越界）
  T2.3 – 數據量不足時 gap 自動縮減；T < n_folds×2 時退化為單折
"""
import sys
sys.path.insert(0, r'd:\cl\MT5_AlphaGPT')

import pytest
from model_core.engine import _build_walk_forward_folds


class TestNormalCase:
    """T2.1 – 正常情況：val_start == train_end + gap（使用折內記錄的實際 gap）"""

    def test_val_start_equals_train_end_plus_gap(self):
        """T=500, n_folds=5, gap=20：每折的 val_start 嚴格等於 train_end + gap。"""
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        assert len(folds) > 0, "應至少返回一折"
        for i, fold in enumerate(folds):
            actual_gap = fold["gap"]
            assert fold["val_start"] == fold["train_end"] + actual_gap, (
                f"折 {i}: val_start={fold['val_start']} != "
                f"train_end={fold['train_end']} + gap={actual_gap}"
            )

    def test_gap_preserved_when_data_sufficient(self):
        """當數據充足時（T mod n_folds >= gap*(n_folds-1)），gap 不被縮減。
        T=53, n_folds=3, gap=1：53 mod 3 = 2 >= 1*(3-1)=2，gap 恰好被保留為 1。
        """
        folds = _build_walk_forward_folds(T=53, n_folds=3, gap=1)
        assert len(folds) > 0, "應返回至少一折"
        for fold in folds:
            assert "gap" in fold, "折字典缺少 'gap' 鍵"
            assert fold["gap"] == 1, f"gap 應保留為 1，實際 gap={fold['gap']}"

    def test_expected_fold_count(self):
        """T=500, n_folds=5, gap=20：應有 4 折（k=1~4）。"""
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        assert len(folds) == 4

    def test_train_start_always_zero(self):
        """所有折的 train_start 都為 0（擴展訓練窗口）。"""
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            assert fold["train_start"] == 0, (
                f"折 {i}: train_start 應為 0，實際為 {fold['train_start']}"
            )

    def test_val_end_within_bounds(self):
        """所有折的 val_end <= T。"""
        T = 500
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            assert fold["val_end"] <= T, (
                f"折 {i}: val_end={fold['val_end']} 超出 T={T}"
            )


class TestGapZero:
    """gap=0 時 val_start 應等於 train_end。"""

    def test_val_start_equals_train_end_when_gap_zero(self):
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=0)
        assert len(folds) > 0
        for i, fold in enumerate(folds):
            assert fold["val_start"] == fold["train_end"], (
                f"折 {i}: gap=0 時 val_start 應等於 train_end，"
                f"實際 val_start={fold['val_start']}, train_end={fold['train_end']}"
            )

    def test_gap_stored_as_zero(self):
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=0)
        for fold in folds:
            assert fold["gap"] == 0


class TestInsufficientData:
    """T2.3 – 數據量不足時 gap 自動縮減，摺疊不越界。"""

    def test_no_fold_has_val_start_beyond_T(self):
        """T=50, n_folds=5, gap=20：gap 應被自動縮減，所有折 val_start < T。"""
        T = 50
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            assert fold["val_start"] < T, (
                f"折 {i}: val_start={fold['val_start']} >= T={T}"
            )

    def test_no_fold_has_val_end_beyond_T(self):
        """T=50, n_folds=5, gap=20：所有折 val_end <= T。"""
        T = 50
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            assert fold["val_end"] <= T, (
                f"折 {i}: val_end={fold['val_end']} > T={T}"
            )

    def test_gap_reduced_when_insufficient_data(self):
        """T=50, n_folds=5, gap=20：實際 gap 應 < 20（已縮減）。"""
        T = 50
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        for fold in folds:
            assert fold["gap"] < 20, (
                f"數據不足時 gap 應縮減，但折 gap={fold['gap']}"
            )

    def test_val_start_still_equals_train_end_plus_actual_gap(self):
        """數據不足時縮減後的 gap 仍滿足 val_start == train_end + actual_gap。"""
        folds = _build_walk_forward_folds(T=50, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            actual_gap = fold["gap"]
            assert fold["val_start"] == fold["train_end"] + actual_gap, (
                f"折 {i}: val_start={fold['val_start']} != "
                f"train_end={fold['train_end']} + gap={actual_gap}"
            )


class TestDegenerateCase:
    """T2.3 – T < n_folds×2 時退化為單折（全量訓練，無驗證）。"""

    def test_returns_single_fold_when_T_too_small(self):
        """T=8, n_folds=5：fold_size=8//5=1 < 2，應退化為單折。"""
        folds = _build_walk_forward_folds(T=8, n_folds=5, gap=20)
        assert len(folds) == 1, (
            f"T=8, n_folds=5 時應退化為單折，實際返回 {len(folds)} 折"
        )

    def test_degenerate_fold_covers_full_range(self):
        """退化單折的 train 和 val 均覆蓋全量 [0, T)。"""
        T = 8
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=20)
        fold = folds[0]
        assert fold["train_start"] == 0
        assert fold["train_end"] == T
        assert fold["val_start"] == 0
        assert fold["val_end"] == T

    def test_degenerate_fold_gap_is_zero(self):
        """退化單折的 gap 為 0（無意義的 gap）。"""
        folds = _build_walk_forward_folds(T=8, n_folds=5, gap=20)
        assert folds[0]["gap"] == 0

    def test_T_equals_n_folds_times_2_minus_1(self):
        """fold_size = (n_folds*2 - 1) // n_folds = 1 < 2，仍退化為單折。"""
        T = 5 * 2 - 1   # = 9, fold_size = 9//5 = 1
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=0)
        assert len(folds) == 1

    def test_T_equals_n_folds_times_2_gives_multiple_folds(self):
        """fold_size = (n_folds*2) // n_folds = 2 >= 2，應產生多折。"""
        T = 5 * 2   # = 10, fold_size = 10//5 = 2
        folds = _build_walk_forward_folds(T=T, n_folds=5, gap=0)
        assert len(folds) > 1, (
            f"fold_size=2 時應產生多折，實際返回 {len(folds)} 折"
        )


class TestReturnStructure:
    """驗證返回的折字典包含所有必要鍵。"""

    REQUIRED_KEYS = {"train_start", "train_end", "val_start", "val_end", "gap"}

    def test_all_required_keys_present(self):
        folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            missing = self.REQUIRED_KEYS - set(fold.keys())
            assert not missing, f"折 {i} 缺少鍵：{missing}"

    def test_degenerate_fold_has_required_keys(self):
        folds = _build_walk_forward_folds(T=8, n_folds=5, gap=20)
        for i, fold in enumerate(folds):
            missing = self.REQUIRED_KEYS - set(fold.keys())
            assert not missing, f"退化折 {i} 缺少鍵：{missing}"
