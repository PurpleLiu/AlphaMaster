from __future__ import annotations

import numpy as np
import pytest

from judgment.core import build_net_returns, cost_stress, performance_metrics


def test_build_net_returns_charges_every_position_change() -> None:
    position = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    market = np.array([0.0, 0.01, -0.005, 0.02, 0.0])

    net, turnover = build_net_returns(position, market, one_way_cost=0.001)

    np.testing.assert_allclose(turnover, [0.0, 1.0, 0.0, 2.0, 1.0])
    np.testing.assert_allclose(net, position * market - turnover * 0.001)


def test_performance_metrics_include_pf_drawdown_and_exposure() -> None:
    pnl = np.array([0.01, -0.005, 0.02, -0.002])
    pos = np.array([1.0, 1.0, -0.5, 0.0])

    metrics = performance_metrics(pnl, pos, periods_per_year=365)

    assert abs(metrics["profit_factor"] - (0.03 / 0.007)) < 1e-12
    assert metrics["total_log_return"] == 0.023
    assert metrics["max_drawdown"] < 0
    assert metrics["exposure_ratio"] == 0.75
    assert metrics["long_exposure_ratio"] == 0.5
    assert metrics["short_exposure_ratio"] == 0.25


def test_performance_metrics_report_turnover_when_supplied() -> None:
    pnl = np.array([0.0, 0.01, -0.002])
    pos = np.array([0.0, 1.0, 0.0])
    turnover = np.array([0.0, 1.0, 1.0])

    metrics = performance_metrics(pnl, pos, 365, turnover=turnover)

    assert metrics["total_turnover"] == 2.0
    assert metrics["turnover_per_bar"] == 2.0 / 3.0
    assert metrics["rebalance_count"] == 2


def test_cost_stress_uses_fixed_multiplier_labels() -> None:
    position = np.array([0.0, 1.0, 1.0, 0.0])
    market = np.array([0.0, 0.01, 0.01, 0.0])

    result = cost_stress(position, market, 0.001, (1, 2, 3, 5), 365)

    assert list(result) == ["1x", "2x", "3x", "5x"]
    assert result["1x"]["one_way_cost"] == 0.001
    assert result["1x"]["total_log_return"] > result["5x"]["total_log_return"]


def test_performance_metrics_reject_non_finite_returns_instead_of_zeroing_them() -> None:
    with pytest.raises(ValueError, match="淨報酬含有非有限值"):
        performance_metrics(
            np.array([0.01, np.nan]), np.array([1.0, 1.0]), periods_per_year=365
        )


def test_build_net_returns_rejects_misaligned_inputs_in_traditional_chinese() -> None:
    with pytest.raises(ValueError, match="部位與市場報酬長度必須一致"):
        build_net_returns(np.array([0.0]), np.array([0.0, 0.01]), 0.001)
