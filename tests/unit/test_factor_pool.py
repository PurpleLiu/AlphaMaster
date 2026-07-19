"""
單元測試：factor_pool 因子去相關池

測試 _update_factor_pool 與 _apply_corr_penalty 的核心行為。
無需 data_manager，直接實例化 AlphaEngine(data_manager=None)。

需求：T3.1~T3.7
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig


# ── 公共 fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """無 data_manager 的 AlphaEngine 實例，供所有測試復用。"""
    return AlphaEngine(data_manager=None)


def _make_factor() -> torch.Tensor:
    """返回形狀 [5, 50] 的隨機因子張量。"""
    return torch.randn(5, 50)


# ── TestUpdateFactorPool ──────────────────────────────────────────────────

class TestUpdateFactorPool:
    """測試 _update_factor_pool 的池容量上限與 Top-K 正確性（T3.1, T3.2）。"""

    def test_pool_size_does_not_exceed_top_k(self, engine):
        """插入 FACTOR_TOP_K + 5 個因子後，池大小恰好等於 FACTOR_TOP_K（T3.1）。"""
        n_insert = ModelConfig.FACTOR_TOP_K + 5
        for i in range(n_insert):
            score = float(i)          # 分數單調遞增，確保每個都被考慮
            engine._update_factor_pool(score, _make_factor())

        assert len(engine.factor_pool) == ModelConfig.FACTOR_TOP_K

    def test_pool_contains_top_k_scores(self, engine):
        """池中保留的是歷史最高 K 個分數；被丟棄的分數均低於池內最小分（T3.2）。"""
        n_insert = ModelConfig.FACTOR_TOP_K + 5
        all_scores = []
        for i in range(n_insert):
            score = float(i)
            all_scores.append(score)
            engine._update_factor_pool(score, _make_factor())

        # 池中分數（堆元素第 0 項）
        pool_scores = sorted(s for s, _cnt, _ in engine.factor_pool)
        # 歷史最高 K 個分數
        expected_top_k = sorted(all_scores)[-ModelConfig.FACTOR_TOP_K:]

        assert pool_scores == expected_top_k

    def test_low_score_rejected_when_pool_full(self, engine):
        """池已滿時，分數不高於堆頂的因子不被入池（T3.2 嚴格大於規則）。"""
        # 先填滿池，分數為 10.0 ~ 10.0+K-1
        for i in range(ModelConfig.FACTOR_TOP_K):
            engine._update_factor_pool(10.0 + i, _make_factor())

        min_pool_score_before = engine.factor_pool[0][0]   # 堆頂（最小分）

        # 嘗試插入低於堆頂的分數
        engine._update_factor_pool(min_pool_score_before - 1.0, _make_factor())

        # 池大小不變，堆頂不變
        assert len(engine.factor_pool) == ModelConfig.FACTOR_TOP_K
        assert engine.factor_pool[0][0] == min_pool_score_before

    def test_equal_score_not_replacing_pool_entry(self, engine):
        """分數等於堆頂時，不替換（嚴格大於才替換，T3.2）。"""
        for i in range(ModelConfig.FACTOR_TOP_K):
            engine._update_factor_pool(5.0 + i, _make_factor())

        heap_top_before = engine.factor_pool[0][0]
        engine._update_factor_pool(heap_top_before, _make_factor())   # 等於堆頂，不替換

        assert len(engine.factor_pool) == ModelConfig.FACTOR_TOP_K
        assert engine.factor_pool[0][0] == heap_top_before

    def test_factor_stored_on_cpu(self, engine):
        """因子張量以 CPU tensor 儲存，節省 VRAM（T3.1）。"""
        engine._update_factor_pool(1.0, _make_factor())
        _, _cnt, stored = engine.factor_pool[0]
        assert stored.device.type == "cpu"


# ── TestApplyCorrPenalty ──────────────────────────────────────────────────

class TestApplyCorrPenalty:
    """測試 _apply_corr_penalty 的各種場景（T3.3~T3.7）。"""

    def test_penalty_applied_for_identical_factors(self, engine):
        """兩個完全相同的因子相關係數為 1.0，超過閾值 0.7，reward 應乘以 CORR_PENALTY（T3.3, T3.4）。"""
        factor = torch.randn(5, 50)
        # 將該因子（相同張量）放入池中，corr = 1.0
        engine._update_factor_pool(1.0, factor)

        reward = torch.tensor(2.0)
        penalized = engine._apply_corr_penalty(reward, factor)

        expected = reward * ModelConfig.CORR_PENALTY
        assert torch.isclose(penalized, expected), (
            f"期望 {expected.item():.4f}，得到 {penalized.item():.4f}"
        )

    def test_penalty_applied_only_once_for_multiple_correlated_pool_entries(self, engine):
        """多個池因子均與候選因子高相關時，懲罰最多施加一次（T3.4）。"""
        factor = torch.randn(5, 50)
        # 放入 3 個與 factor 完全相同的因子（使用不同分數避免堆比較 tensor）
        for i in range(3):
            engine._update_factor_pool(1.0 + i * 0.1, factor.clone())

        reward = torch.tensor(4.0)
        penalized = engine._apply_corr_penalty(reward, factor)

        # 只懲罰一次：4.0 * 0.5 = 2.0，而非 4.0 * 0.5^3 = 0.5
        expected = reward * ModelConfig.CORR_PENALTY
        assert torch.isclose(penalized, expected), (
            f"懲罰應僅施加一次，期望 {expected.item():.4f}，得到 {penalized.item():.4f}"
        )

    def test_no_penalty_for_uncorrelated_factor(self, engine):
        """與池中因子相關係數低於閾值時，reward 不變（T3.3, T3.4）。"""
        pool_factor = torch.zeros(5, 50)
        pool_factor[0, 0] = 1.0   # 近似常數，corr 極低
        # 構造與 pool_factor 完全無關的因子
        candidate = torch.randn(5, 50)
        # 強制讓 candidate 與 pool_factor 正交（減去投影）
        pf_flat = pool_factor.reshape(-1).float()
        c_flat = candidate.reshape(-1).float()
        proj = (c_flat @ pf_flat) / (pf_flat @ pf_flat + 1e-8) * pf_flat
        c_ortho = c_flat - proj
        # 正交化後 std 可能足夠大
        if c_ortho.std() > 1e-4:
            engine._update_factor_pool(1.0, pool_factor)
            reward = torch.tensor(3.0)
            # 計算實際相關係數
            f_c = c_ortho - c_ortho.mean()
            p_c = pf_flat - pf_flat.mean()
            corr = (f_c @ p_c) / (f_c.norm() * p_c.norm() + 1e-8)
            if corr.abs() <= ModelConfig.CORR_THRESHOLD:
                candidate_tensor = c_ortho.reshape(5, 50)
                result = engine._apply_corr_penalty(reward, candidate_tensor)
                assert torch.isclose(result, reward), (
                    f"低相關因子不應被懲罰，期望 {reward.item():.4f}，得到 {result.item():.4f}"
                )

    def test_constant_factor_skips_penalty(self, engine):
        """std < 1e-4 的常數因子跳過懲罰，reward 不變（T3.7）。"""
        # 先向池中插入一個正常因子（corr 會是 1.0 若常數，但應在 std 檢測前返回）
        pool_factor = torch.randn(5, 50)
        engine._update_factor_pool(1.0, pool_factor)

        # 常數因子：全為同一值
        constant_factor = torch.full((5, 50), 3.14)
        reward = torch.tensor(2.5)
        result = engine._apply_corr_penalty(reward, constant_factor)

        assert torch.isclose(result, reward), (
            f"常數因子（std < 1e-4）應跳過懲罰，期望 {reward.item():.4f}，得到 {result.item():.4f}"
        )

    def test_empty_pool_returns_reward_unchanged(self, engine):
        """池為空時直接返回原始 reward，無任何修改（T3.6）。"""
        assert len(engine.factor_pool) == 0, "初始池應為空"

        reward = torch.tensor(1.5)
        factor = torch.randn(5, 50)
        result = engine._apply_corr_penalty(reward, factor)

        assert torch.isclose(result, reward), (
            f"空池時 reward 應不變，期望 {reward.item():.4f}，得到 {result.item():.4f}"
        )

    def test_corr_penalty_value_is_config_value(self, engine):
        """懲罰係數嚴格等於 ModelConfig.CORR_PENALTY（T3.5）。"""
        factor = torch.randn(5, 50)
        engine._update_factor_pool(1.0, factor)

        reward = torch.tensor(1.0)
        result = engine._apply_corr_penalty(reward, factor)

        assert float(result) == pytest.approx(ModelConfig.CORR_PENALTY), (
            f"懲罰後值應為 CORR_PENALTY={ModelConfig.CORR_PENALTY}，得到 {float(result):.6f}"
        )

    def test_reward_is_tensor_after_penalty(self, engine):
        """懲罰後返回值仍為 tensor，類型未改變。"""
        factor = torch.randn(5, 50)
        engine._update_factor_pool(1.0, factor)

        reward = torch.tensor(2.0)
        result = engine._apply_corr_penalty(reward, factor)

        assert isinstance(result, torch.Tensor)
