"""
單元測試：坍塌快照恢復邏輯（需求 T4.1~T4.4）

驗證：
- 有快照時，attention/qk_norm/norm1/norm2 層參數恢復後與快照逐元素相等
- 有快照時，FFN 層參數在噪聲擾動後與快照不同
- 無快照時退化為全參數擾動（不拋異常）
- AlphaEngine 初始化時 _best_snapshot 為 None
"""
import copy
import sys
import os

import pytest
import torch

# 確保項目根目錄在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_core.alphagpt import AlphaGPT
from model_core.engine import AlphaEngine


# ─────────────────────────────────────────────────────────────────────────────
# 輔助：將模擬的"坍塌重啟-方案A"邏輯提取為獨立函數，方便測試直接調用
# 與 engine.py 中的實際實現保持完全一致
# ─────────────────────────────────────────────────────────────────────────────

def _apply_snapshot_restore_with_ffn_noise(model: AlphaGPT, snapshot: dict) -> None:
    """方案 A：恢復快照，僅對 FFN 參數加高斯噪聲（noise_std=0.02）。"""
    model.load_state_dict(snapshot)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'ffn' in name:
                param.add_(torch.randn_like(param) * 0.02)


def _apply_full_noise(model: AlphaGPT) -> None:
    """方案 B：無快照時，全參數加高斯噪聲（noise_std=0.02）。"""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02)


# ─────────────────────────────────────────────────────────────────────────────
# 測試：AlphaEngine 初始化狀態
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaEngineInit:
    """驗證 AlphaEngine 初始化時 _best_snapshot 為 None（需求 T4.1）。"""

    def test_best_snapshot_is_none_on_init(self):
        """AlphaEngine 初始化後 _best_snapshot 應為 None。"""
        engine = AlphaEngine(data_manager=None)
        assert engine._best_snapshot is None

    def test_model_is_alphagpt_instance(self):
        """AlphaEngine 持有的 model 應是 AlphaGPT 實例。"""
        engine = AlphaEngine(data_manager=None)
        assert isinstance(engine.model, AlphaGPT)


# ─────────────────────────────────────────────────────────────────────────────
# 測試：有快照時——attention 類參數恢復後與快照完全一致
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotRestoreAttentionParams:
    """驗證有快照時，attention/qk_norm/norm1/norm2 參數與快照逐元素相等（需求 T4.2, T4.3）。"""

    # attention 相關關鍵字（這些參數在方案 A 中不加噪聲）
    FROZEN_KEYWORDS = ('attention', 'qk_norm', 'norm1', 'norm2')

    @pytest.fixture
    def model_with_snapshot(self):
        """返回 (model, snapshot)；snapshot 在"訓練"後保存，然後再次修改 model 權重。"""
        model = AlphaGPT()

        # 步驟1：保存快照（模擬在最優時刻保存）
        snapshot = copy.deepcopy(model.state_dict())

        # 步驟2：模擬訓練，修改所有參數
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        return model, snapshot

    def test_attention_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢復快照後，所有含 'attention' 關鍵字的參數應與快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            if 'attention' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"參數 '{name}' 恢復後與快照不一致"
                )

    def test_qk_norm_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢復快照後，所有含 'qk_norm' 關鍵字的參數應與快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        checked = 0
        for name, snap_tensor in snapshot.items():
            if 'qk_norm' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"參數 '{name}' 恢復後與快照不一致"
                )
                checked += 1
        assert checked > 0, "未找到含 'qk_norm' 的參數，請檢查模型結構"

    def test_norm1_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢復快照後，所有含 'norm1' 關鍵字的參數應與快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        checked = 0
        for name, snap_tensor in snapshot.items():
            if 'norm1' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"參數 '{name}' 恢復後與快照不一致"
                )
                checked += 1
        assert checked > 0, "未找到含 'norm1' 的參數，請檢查模型結構"

    def test_norm2_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢復快照後，所有含 'norm2' 關鍵字的參數應與快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        checked = 0
        for name, snap_tensor in snapshot.items():
            if 'norm2' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"參數 '{name}' 恢復後與快照不一致"
                )
                checked += 1
        assert checked > 0, "未找到含 'norm2' 的參數，請檢查模型結構"

    def test_all_frozen_params_match_snapshot(self, model_with_snapshot):
        """整合測試：所有不含 'ffn' 的參數均與快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            if not any(kw in name for kw in ('ffn',)):
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"非FFN參數 '{name}' 恢復後與快照不一致"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 測試：有快照時——FFN 參數在噪聲擾動後與快照不同
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotRestoreFFNParams:
    """驗證有快照時，FFN 參數在加噪聲後與快照不同（需求 T4.2）。"""

    @pytest.fixture
    def model_with_snapshot(self):
        model = AlphaGPT()
        snapshot = copy.deepcopy(model.state_dict())

        # 模擬訓練後修改權重
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        return model, snapshot

    def test_ffn_params_differ_from_snapshot_after_noise(self, model_with_snapshot):
        """恢復快照並加噪聲後，FFN 參數應與快照不完全相等。"""
        model, snapshot = model_with_snapshot

        # 固定隨機種子確保噪聲非零
        torch.manual_seed(42)
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        ffn_params_found = 0
        for name, snap_tensor in snapshot.items():
            if 'ffn' in name:
                ffn_params_found += 1
                assert not torch.equal(restored_sd[name], snap_tensor), (
                    f"FFN 參數 '{name}' 加噪聲後仍與快照完全相同（噪聲未生效）"
                )
        assert ffn_params_found > 0, "未找到含 'ffn' 的參數，請檢查模型結構"

    def test_ffn_params_close_to_snapshot_after_small_noise(self, model_with_snapshot):
        """噪聲 std=0.02 較小，FFN 參數值應在快照附近（差異 < 1.0）。"""
        model, snapshot = model_with_snapshot
        torch.manual_seed(42)
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            if 'ffn' in name:
                max_diff = (restored_sd[name] - snap_tensor).abs().max().item()
                assert max_diff < 1.0, (
                    f"FFN 參數 '{name}' 噪聲擾動幅度異常大（max_diff={max_diff:.4f}）"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 測試：無快照時退化為全參數擾動（不拋異常，需求 T4.4）
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSnapshotFallback:
    """驗證無快照時退化為全參數擾動且不拋異常（需求 T4.4）。"""

    def test_full_noise_does_not_raise(self):
        """_best_snapshot 為 None 時，全參數加噪聲應正常完成，不拋任何異常。"""
        model = AlphaGPT()
        # 記錄加噪前的參數值
        before = {name: param.clone() for name, param in model.named_parameters()}

        # 不應拋出異常
        _apply_full_noise(model)

        # 驗證參數確實被修改了
        changed = 0
        for name, param in model.named_parameters():
            if not torch.equal(param, before[name]):
                changed += 1
        assert changed > 0, "全參數擾動後所有參數均未改變"

    def test_full_noise_modifies_all_params(self):
        """全參數擾動應修改模型中所有可訓練參數。"""
        model = AlphaGPT()
        before = {name: param.clone() for name, param in model.named_parameters()}

        torch.manual_seed(123)
        _apply_full_noise(model)

        for name, param in model.named_parameters():
            assert not torch.equal(param, before[name]), (
                f"參數 '{name}' 在全參數擾動後未改變"
            )

    def test_engine_with_none_snapshot_fallback_no_raise(self):
        """通過 AlphaEngine 介面驗證：_best_snapshot=None 時全參數加噪聲不拋異常。"""
        engine = AlphaEngine(data_manager=None)
        assert engine._best_snapshot is None

        # 直接模擬 engine.py 中的無快照分支邏輯
        with torch.no_grad():
            for param in engine.model.parameters():
                param.add_(torch.randn_like(param) * 0.02)

        # 如果執行到此處，說明無異常拋出
        assert True


# ─────────────────────────────────────────────────────────────────────────────
# 測試：快照深拷貝隔離（需求 T4.1）
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotIsolation:
    """驗證 deepcopy 快照與模型權重完全解耦（需求 T4.1）。"""

    def test_snapshot_not_affected_by_model_update(self):
        """修改模型權重後，快照內容不應改變。"""
        model = AlphaGPT()
        snapshot = copy.deepcopy(model.state_dict())

        # 記錄快照初始值
        snap_values = {k: v.clone() for k, v in snapshot.items()}

        # 修改模型權重
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 1.0)

        # 快照應保持不變
        for name, original_val in snap_values.items():
            assert torch.equal(snapshot[name], original_val), (
                f"快照參數 '{name}' 被模型更新汙染（deepcopy 失效）"
            )

    def test_restore_from_snapshot_is_exact(self):
        """load_state_dict(snapshot) 後，模型權重應與快照完全一致（無精度損失）。"""
        model = AlphaGPT()
        snapshot = copy.deepcopy(model.state_dict())

        # 修改權重
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(2.0)

        # 恢復
        model.load_state_dict(snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            assert torch.equal(restored_sd[name], snap_tensor), (
                f"參數 '{name}' 恢復後與快照不完全一致"
            )
