from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from judgment.core import (
    JudgmentThresholds,
    build_net_returns,
    classify_judgment,
    concentration_analysis,
    cost_stress,
    market_regression,
    performance_metrics,
    pseudo_walk_forward,
    temporal_blocks,
)


def _passing_inputs() -> tuple[dict, dict, dict, dict, dict]:
    return (
        {"profit_factor": 1.5, "sharpe": 1.0},
        {"2x": {"total_log_return": 0.1}},
        {"equal": {"median_return": 0.02, "positive_ratio": 0.75}},
        {"annual_alpha": 0.1},
        {"severe": False},
    )


def test_classification_pass_requires_every_gate() -> None:
    result = classify_judgment(*_passing_inputs())

    assert result["status"] == "PASS"
    assert all(rule["passed"] is True for rule in result["rules"])


def test_classification_fail_for_negative_economic_edge() -> None:
    args = list(_passing_inputs())
    args[0] = {"profit_factor": 0.8, "sharpe": -0.2}

    result = classify_judgment(*args)

    assert result["status"] == "FAIL"


def test_classification_review_when_metric_is_unavailable() -> None:
    args = list(_passing_inputs())
    args[3] = {"annual_alpha": None, "reason": "資料不足"}

    result = classify_judgment(*args)

    assert result["status"] == "REVIEW"


@pytest.mark.parametrize(
    ("metric_group", "metric_values"),
    [
        ("full_metrics", {"profit_factor": 0.99, "sharpe": 1.0}),
        ("full_metrics", {"profit_factor": 1.5, "sharpe": 0.0}),
        ("stress", {"2x": {"total_log_return": -0.001}}),
    ],
)
def test_classification_fails_for_each_independent_economic_trigger(
    metric_group: str, metric_values: dict
) -> None:
    full_metrics, stress, blocks, regression, concentration = _passing_inputs()
    inputs = {
        "full_metrics": full_metrics,
        "stress": stress,
        "blocks": blocks,
        "regression": regression,
        "concentration": concentration,
    }
    inputs[metric_group] = metric_values

    result = classify_judgment(**inputs)

    assert result["status"] == "FAIL"


def test_classification_uses_documented_equality_boundaries() -> None:
    result = classify_judgment(
        {"profit_factor": 1.30, "sharpe": 0.75},
        {"2x": {"total_log_return": 0.0}},
        {"equal": {"median_return": 0.0, "positive_ratio": 0.60}},
        {"annual_alpha": 0.0},
        {"severe": False},
    )

    rules = {rule["name"]: rule for rule in result["rules"]}
    assert result["status"] == "REVIEW"
    assert rules["profit_factor"]["passed"] is True
    assert rules["sharpe"]["passed"] is True
    assert rules["cost_stress_2x"]["passed"] is True
    assert rules["positive_block_ratio"]["passed"] is True
    assert rules["block_median"]["passed"] is False
    assert rules["annual_alpha"]["passed"] is False


@pytest.mark.parametrize(
    ("blocks", "regression"),
    [
        ({"equal": {"median_return": 0.0, "positive_ratio": 0.75}}, {"annual_alpha": 0.1}),
        ({"equal": {"median_return": 0.02, "positive_ratio": 0.59}}, {"annual_alpha": 0.1}),
        ({"equal": {"median_return": 0.02, "positive_ratio": 0.75}}, {"annual_alpha": 0.0}),
    ],
)
def test_classification_reviews_ordinary_failed_gates(
    blocks: dict, regression: dict
) -> None:
    full_metrics, stress, _, _, concentration = _passing_inputs()

    result = classify_judgment(full_metrics, stress, blocks, regression, concentration)

    assert result["status"] == "REVIEW"


@pytest.mark.parametrize("concentration", [{"severe": True}, {}, {"severe": None}, None])
def test_classification_reviews_severe_or_unavailable_concentration(
    concentration: dict | None,
) -> None:
    full_metrics, stress, blocks, regression, _ = _passing_inputs()

    result = classify_judgment(full_metrics, stress, blocks, regression, concentration)

    concentration_rule = next(rule for rule in result["rules"] if rule["name"] == "concentration")
    assert result["status"] == "REVIEW"
    if concentration in ({}, {"severe": None}, None):
        assert concentration_rule["passed"] is None
        assert "無法取得" in concentration_rule["explanation"]
    else:
        assert concentration_rule["passed"] is False


def test_classification_uses_custom_thresholds_in_every_rule_explanation() -> None:
    thresholds = JudgmentThresholds(1.25, 0.80, 0.01, 0.02, 0.70, 0.03)
    full_metrics, stress, blocks, regression, concentration = _passing_inputs()
    blocks = {"equal": {"median_return": 0.03, "positive_ratio": 0.75}}

    result = classify_judgment(
        full_metrics, stress, blocks, regression, concentration, thresholds=thresholds
    )

    assert result["status"] == "PASS"
    expected = {
        "profit_factor": ">= 1.25",
        "sharpe": ">= 0.80",
        "cost_stress_2x": ">= 0.01",
        "block_median": "> 0.02",
        "positive_block_ratio": ">= 70%",
        "annual_alpha": "> 0.03",
        "concentration": "== True",
    }
    for rule in result["rules"]:
        assert expected[rule["name"]] in rule["explanation"]
        assert "需" in rule["explanation"]


def test_classification_rule_schema_and_explanations_are_traditional_chinese() -> None:
    result = classify_judgment(*_passing_inputs())

    assert {rule["name"] for rule in result["rules"]} == {
        "profit_factor",
        "sharpe",
        "cost_stress_2x",
        "block_median",
        "positive_block_ratio",
        "annual_alpha",
        "concentration",
    }
    for rule in result["rules"]:
        assert set(rule) == {"name", "value", "threshold", "operator", "passed", "explanation"}
        assert rule["operator"] in {">=", ">", "=="}
        assert "需" in rule["explanation"]


@pytest.mark.parametrize(
    ("stress", "blocks", "regression", "concentration", "unavailable_rule"),
    [
        ({"2x": None}, {"equal": {"median_return": 0.02, "positive_ratio": 0.75}}, {"annual_alpha": 0.1}, {"severe": False}, "cost_stress_2x"),
        ({"2x": {"total_log_return": "0.1"}}, {"equal": {"median_return": 0.02, "positive_ratio": 0.75}}, {"annual_alpha": 0.1}, {"severe": False}, "cost_stress_2x"),
        ({"2x": {"total_log_return": np.nan}}, {"equal": {"median_return": 0.02, "positive_ratio": 0.75}}, {"annual_alpha": 0.1}, {"severe": False}, "cost_stress_2x"),
        ({"2x": {"total_log_return": 0.1}}, {"equal": []}, None, {"severe": False}, "block_median"),
    ],
)
def test_classification_reviews_malformed_or_unavailable_nested_inputs(
    stress: dict,
    blocks: dict,
    regression: dict | None,
    concentration: dict,
    unavailable_rule: str,
) -> None:
    full_metrics, _, _, _, _ = _passing_inputs()

    result = classify_judgment(full_metrics, stress, blocks, regression, concentration)

    rule = next(rule for rule in result["rules"] if rule["name"] == unavailable_rule)
    assert result["status"] == "REVIEW"
    assert rule["passed"] is None
    assert "無法取得" in rule["explanation"]


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


@pytest.mark.parametrize("periods_per_year", [np.nan, np.inf, -np.inf])
def test_performance_metrics_rejects_non_finite_periods_per_year(
    periods_per_year: float,
) -> None:
    with pytest.raises(ValueError, match="期間數"):
        performance_metrics(np.array([0.01]), np.array([1.0]), periods_per_year)


@pytest.mark.parametrize(
    ("position", "market_return", "one_way_cost"),
    [
        (np.array([np.nan]), np.array([0.01]), 0.001),
        (np.array([1.0]), np.array([np.inf]), 0.001),
        (np.array([1.0]), np.array([0.01]), np.nan),
    ],
)
def test_build_net_returns_rejects_non_finite_inputs(
    position: np.ndarray, market_return: np.ndarray, one_way_cost: float
) -> None:
    with pytest.raises(ValueError, match="有限"):
        build_net_returns(position, market_return, one_way_cost)


@pytest.mark.parametrize("turnover", [np.array([np.nan]), np.array([np.inf])])
def test_performance_metrics_rejects_non_finite_turnover(turnover: np.ndarray) -> None:
    with pytest.raises(ValueError, match="有限"):
        performance_metrics(
            np.array([0.01]), np.array([1.0]), periods_per_year=365, turnover=turnover
        )


def test_performance_metrics_rejects_non_finite_position() -> None:
    with pytest.raises(ValueError, match="有限"):
        performance_metrics(np.array([0.01]), np.array([np.nan]), periods_per_year=365)


@pytest.mark.parametrize(
    ("base_cost", "multipliers"),
    [(np.nan, (1.0,)), (0.001, (np.inf,))],
)
def test_cost_stress_rejects_non_finite_cost_inputs(
    base_cost: float, multipliers: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="有限"):
        cost_stress(
            np.array([0.0]),
            np.array([0.0]),
            base_cost=base_cost,
            multipliers=multipliers,
            periods_per_year=365,
        )


def test_build_net_returns_rejects_overflowed_net_returns() -> None:
    with pytest.raises(ValueError, match="計算結果"):
        build_net_returns(np.array([1e308]), np.array([1e308]), 0.0)


def test_performance_metrics_rejects_overflowed_derived_metrics() -> None:
    with pytest.raises(ValueError, match="計算結果"):
        performance_metrics(np.array([1e308, 1e308]), np.array([1.0, 1.0]), 365)


def test_cost_stress_rejects_overflowed_scenario_cost() -> None:
    with pytest.raises(ValueError, match="計算結果"):
        cost_stress(
            np.array([0.0]),
            np.array([0.0]),
            base_cost=1e308,
            multipliers=(1e308,),
            periods_per_year=365,
        )


def test_unavailable_metrics_are_none_with_traditional_chinese_reasons() -> None:
    metrics = performance_metrics(
        np.array([0.01, 0.01]), np.array([1.0, 1.0]), periods_per_year=365
    )

    for metric in ("sharpe", "sortino", "profit_factor"):
        assert metrics[metric] is None
        assert metric in metrics["unavailable_reasons"]
        assert "\u4e00" <= metrics["unavailable_reasons"][metric][0] <= "\u9fff"


def test_temporal_blocks_report_median_and_positive_ratio() -> None:
    times = np.arange(8, dtype=np.int64) * 3600 + 1_600_000_000
    pnl = np.array([0.01, 0.01, -0.01, -0.01, 0.02, 0.02, 0.01, 0.01])

    result = temporal_blocks(pnl, times, 8760, equal_blocks=4)

    assert result["equal"]["returns"] == [0.02, -0.02, 0.04, 0.02]
    assert result["equal"]["median_return"] == 0.02
    assert result["equal"]["positive_ratio"] == 0.75


def test_temporal_blocks_aggregate_natural_calendar_years() -> None:
    times = np.array(
        [
            pd.Timestamp("2024-12-31T23:00:00Z").timestamp(),
            pd.Timestamp("2025-01-01T00:00:00Z").timestamp(),
            pd.Timestamp("2025-01-01T01:00:00Z").timestamp(),
        ]
    )

    result = temporal_blocks(np.array([0.01, -0.02, 0.03]), times, 8760, equal_blocks=2)

    assert [block["year"] for block in result["natural_year"]["blocks"]] == [2024, 2025]
    assert [block["return"] for block in result["natural_year"]["blocks"]] == pytest.approx(
        [0.01, 0.01]
    )


def test_concentration_flags_single_block_dependency() -> None:
    result = concentration_analysis([0.60, -0.05, -0.05, -0.05])

    assert result["severe"] is True
    assert result["return_without_best_block"] < 0


@pytest.mark.parametrize("block_returns", [[0.1, np.nan], [0.1, np.inf]])
def test_concentration_rejects_non_finite_block_returns(block_returns: list[float]) -> None:
    with pytest.raises(ValueError, match="非有限"):
        concentration_analysis(block_returns)


def test_market_regression_recovers_beta_and_positive_alpha() -> None:
    benchmark = np.linspace(-0.01, 0.01, 200)
    strategy = 0.5 * benchmark + 0.0002 + np.sin(np.arange(200)) * 1e-5

    result = market_regression(strategy, benchmark, periods_per_year=365)

    assert abs(result["beta"] - 0.5) < 1e-3
    assert result["annual_alpha"] > 0
    assert result["correlation"] > 0.99
    assert result["residual_sharpe"] is not None


def test_market_regression_filters_non_finite_pairs_before_fitting() -> None:
    benchmark = np.linspace(-0.01, 0.01, 32)
    strategy = 0.5 * benchmark + 0.0002 + np.sin(np.arange(32)) * 1e-5
    strategy[0] = np.nan
    benchmark[1] = np.inf

    result = market_regression(strategy, benchmark, periods_per_year=365)

    assert result["observations"] == 30
    assert result["beta"] == pytest.approx(0.5, abs=1e-3)


def test_market_regression_requires_at_least_thirty_finite_pairs() -> None:
    result = market_regression(
        np.r_[np.full(29, 0.001), np.nan],
        np.r_[np.linspace(-0.01, 0.01, 29), 0.001],
        periods_per_year=365,
    )

    assert result["observations"] == 29
    assert all(result[key] is None for key in ("alpha", "annual_alpha", "beta", "residual_sharpe", "correlation"))
    assert result["reason"] and "\u4e00" <= result["reason"][0] <= "\u9fff"


@pytest.mark.parametrize(
    ("strategy", "benchmark"),
    [
        (np.full(30, 0.001), np.linspace(-0.01, 0.01, 30)),
        (np.linspace(-0.01, 0.01, 30), np.full(30, 0.001)),
    ],
)
def test_market_regression_returns_unavailable_for_constant_series(
    strategy: np.ndarray, benchmark: np.ndarray
) -> None:
    result = market_regression(strategy, benchmark, periods_per_year=365)

    assert all(result[key] is None for key in ("alpha", "annual_alpha", "beta", "residual_sharpe", "correlation"))
    assert result["reason"] and "報酬" in result["reason"]


def test_market_regression_returns_unavailable_for_zero_residual_standard_deviation() -> None:
    benchmark = np.linspace(-0.01, 0.01, 30)
    result = market_regression(0.5 * benchmark + 0.0002, benchmark, periods_per_year=365)

    assert all(result[key] is None for key in ("alpha", "annual_alpha", "beta", "residual_sharpe", "correlation"))
    assert result["reason"] and "殘差" in result["reason"]


def test_pseudo_walk_forward_applies_embargo_and_marks_limitation() -> None:
    pnl = np.full(100, 0.001)
    times = np.arange(100, dtype=np.int64) * 3600 + 1_600_000_000

    result = pseudo_walk_forward(pnl, times, 8760, splits=5, embargo_bars=2)

    assert len(result["folds"]) == 5
    assert result["folds"][1]["start_index"] == 22
    assert result["is_true_out_of_sample"] is False
    """
    assert result["limitation"] == "歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。"


    """
    assert result["limitation"] == "\u6b77\u53f2\u8cc7\u6599\u66fe\u53c3\u8207\u7b56\u7565\u641c\u5c0b\uff1b\u6b64\u7d50\u679c\u50c5\u70ba\u507d\u6a23\u672c\u5916\u9a57\u8b49\u3002"


def test_pseudo_walk_forward_reports_fully_embargoed_fold() -> None:
    pnl = np.full(4, 0.001)
    times = np.arange(4, dtype=np.int64) * 3600 + 1_600_000_000

    result = pseudo_walk_forward(pnl, times, 8760, splits=2, embargo_bars=2)

    assert result["folds"][1]["metrics"] is None
    """
    """
    assert result["folds"][1]["reason"] == "\u9694\u96e2\u671f\u5f8c\u6c92\u6709\u53ef\u8a55\u4f30\u7684\u8cc7\u6599"
