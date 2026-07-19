from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JudgmentThresholds:
    """策略判斷的公開、可調整門檻。"""

    min_profit_factor: float = 1.30
    min_sharpe: float = 0.75
    min_stress_2x_return: float = 0.0
    min_block_median: float = 0.0
    min_positive_block_ratio: float = 0.60
    min_annual_alpha: float = 0.0


def _is_available_judgment_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _judgment_rule(
    name: str, value: Any, threshold: Any, operator: str, explanation: str
) -> dict[str, Any]:
    if value is None or (name != "concentration" and not _is_available_judgment_value(value)):
        passed: bool | None = None
        explanation = f"{explanation}；目前無可用數值，需人工覆核。"
    elif operator == ">=":
        passed = value >= threshold
    elif operator == ">":
        passed = value > threshold
    else:
        passed = value == threshold

    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "operator": operator,
        "passed": passed,
        "explanation": explanation,
    }


def classify_judgment(
    full_metrics: dict[str, Any],
    stress: dict[str, Any],
    blocks: dict[str, Any],
    regression: dict[str, Any],
    concentration: dict[str, Any],
    thresholds: JudgmentThresholds | None = None,
) -> dict[str, Any]:
    """依公開門檻將策略結果分類為 PASS、REVIEW 或 FAIL。"""
    t = thresholds or JudgmentThresholds()
    checks = [
        (
            "profit_factor",
            full_metrics.get("profit_factor"),
            t.min_profit_factor,
            ">=",
            f"獲利因子必須大於或等於 {t.min_profit_factor:.2f}",
        ),
        (
            "sharpe",
            full_metrics.get("sharpe"),
            t.min_sharpe,
            ">=",
            f"夏普比率必須大於或等於 {t.min_sharpe:.2f}",
        ),
        (
            "cost_stress_2x",
            stress.get("2x", {}).get("total_log_return"),
            t.min_stress_2x_return,
            ">=",
            "兩倍交易成本壓力測試的總對數報酬必須非負",
        ),
        (
            "block_median",
            blocks.get("equal", {}).get("median_return"),
            t.min_block_median,
            ">",
            "等分期間報酬中位數必須為正",
        ),
        (
            "positive_block_ratio",
            blocks.get("equal", {}).get("positive_ratio"),
            t.min_positive_block_ratio,
            ">=",
            f"正報酬等分期間比例必須大於或等於 {t.min_positive_block_ratio:.0%}",
        ),
        (
            "annual_alpha",
            regression.get("annual_alpha"),
            t.min_annual_alpha,
            ">",
            "年度 Alpha 必須為正",
        ),
        (
            "concentration",
            not concentration.get("severe") if concentration.get("severe") is not None else None,
            True,
            "==",
            "報酬不可嚴重集中於少數期間",
        ),
    ]
    rules = [_judgment_rule(*check) for check in checks]

    if any(rule["passed"] is None for rule in rules):
        status = "REVIEW"
    elif (
        full_metrics.get("profit_factor") < 1.0
        or full_metrics.get("sharpe") <= 0
        or stress.get("2x", {}).get("total_log_return") < 0
    ):
        status = "FAIL"
    elif all(rule["passed"] for rule in rules):
        status = "PASS"
    else:
        status = "REVIEW"

    return {"status": status, "rules": rules}


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name}含有非有限值，無法進行計算")
    return vector


def _as_positive_finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必須是大於零的有限數值") from error
    if not math.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name}必須是大於零的有限數值")
    return scalar


def _as_non_negative_finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必須是大於或等於零的有限數值") from error
    if not math.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{name}必須是大於或等於零的有限數值")
    return scalar


def _as_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}必須是正整數")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必須是正整數") from error
    if integer <= 0 or integer != value:
        raise ValueError(f"{name}必須是正整數")
    return integer


def _time_index(times: np.ndarray, expected_length: int) -> pd.DatetimeIndex:
    seconds = _as_finite_vector(times, "時間戳記")
    if len(seconds) != expected_length:
        raise ValueError("報酬與時間戳記長度必須一致")
    if expected_length == 0:
        raise ValueError("報酬與時間戳記不得為空")
    try:
        return pd.DatetimeIndex(pd.to_datetime(seconds, unit="s", utc=True))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("時間戳記無法轉換為 UTC 時間") from error


def _unavailable_regression(reason: str, observations: int) -> dict[str, Any]:
    return {
        "observations": observations,
        "alpha": None,
        "annual_alpha": None,
        "beta": None,
        "residual_sharpe": None,
        "correlation": None,
        "reason": reason,
    }


def _require_finite_array(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name}計算結果出現非有限數值")
    return values


def _require_finite_scalar(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name}計算結果出現非有限數值")
    return value


def build_net_returns(
    position: np.ndarray,
    market_return: np.ndarray,
    one_way_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """依部位變動扣除單邊成本，並回傳淨報酬與換手。"""
    pos = _as_finite_vector(position, "部位")
    ret = _as_finite_vector(market_return, "市場報酬")
    cost = _as_non_negative_finite_scalar(one_way_cost, "單邊成本")
    if pos.shape != ret.shape:
        raise ValueError("部位與市場報酬長度必須一致")

    with np.errstate(over="ignore", invalid="ignore"):
        previous = np.concatenate(([0.0], pos[:-1]))
        turnover = np.abs(pos - previous)
        gross_returns = pos * ret
        cost_returns = turnover * cost
        net_returns = gross_returns - cost_returns
    _require_finite_array(turnover, "換手")
    _require_finite_array(gross_returns, "毛報酬")
    _require_finite_array(cost_returns, "成本報酬")
    return _require_finite_array(net_returns, "淨報酬"), turnover


def performance_metrics(
    net_returns: np.ndarray,
    position: np.ndarray,
    periods_per_year: int,
    turnover: np.ndarray | None = None,
) -> dict[str, Any]:
    """計算淨對數報酬的績效指標，並明確標示不可用的指標。"""
    pnl = _as_finite_vector(net_returns, "淨報酬")
    pos = _as_finite_vector(position, "部位")
    periods = _as_positive_finite_scalar(periods_per_year, "每年期間數")
    if len(pnl) == 0:
        raise ValueError("淨報酬與部位不可為空")
    if len(pnl) != len(pos):
        raise ValueError("淨報酬與部位長度必須一致")

    turn = None
    if turnover is not None:
        turn = _as_finite_vector(turnover, "換手")
        if len(turn) != len(pnl):
            raise ValueError("換手與淨報酬長度必須一致")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = _require_finite_scalar(float(pnl.mean()), "平均淨報酬")
        std = _require_finite_scalar(float(pnl.std(ddof=0)), "淨報酬標準差")
        downside = pnl[pnl < 0]
        downside_std = (
            _require_finite_scalar(float(downside.std(ddof=0)), "下行報酬標準差")
            if len(downside)
            else None
        )
        curve = _require_finite_array(np.cumsum(pnl), "累積報酬")
        peaks = _require_finite_array(
            np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:], "累積報酬峰值"
        )
        drawdown = _require_finite_array(curve - peaks, "回撤")
        gains = _require_finite_scalar(float(pnl[pnl > 0].sum()), "獲利總和")
        losses = _require_finite_scalar(float(-pnl[pnl < 0].sum()), "虧損總和")
        total_log = _require_finite_scalar(float(pnl.sum()), "總對數報酬")
        years = _require_finite_scalar(len(pnl) / periods, "年數")
        annual_log = _require_finite_scalar(total_log / years, "年化對數報酬")
        total_return = _require_finite_scalar(float(np.expm1(total_log)), "總報酬")
        annual_return = _require_finite_scalar(float(np.expm1(annual_log)), "年化報酬")

        sharpe = mean / std * math.sqrt(periods) if std > 1e-12 else None
        sortino = (
            mean / downside_std * math.sqrt(periods)
            if downside_std is not None and downside_std > 1e-12
            else None
        )
        profit_factor = gains / losses if losses > 1e-12 else None

        exposure_ratio = float(np.mean(np.abs(pos) > 1e-12))
        mean_absolute_position = float(np.mean(np.abs(pos)))
        long_exposure_ratio = float(np.mean(pos > 1e-12))
        short_exposure_ratio = float(np.mean(pos < -1e-12))

    if sharpe is not None:
        sharpe = _require_finite_scalar(float(sharpe), "夏普比率")
    if sortino is not None:
        sortino = _require_finite_scalar(float(sortino), "索提諾比率")
    if profit_factor is not None:
        profit_factor = _require_finite_scalar(float(profit_factor), "獲利因子")
    exposure_ratio = _require_finite_scalar(exposure_ratio, "曝險比例")
    mean_absolute_position = _require_finite_scalar(
        mean_absolute_position, "平均絕對部位"
    )
    long_exposure_ratio = _require_finite_scalar(long_exposure_ratio, "多頭曝險比例")
    short_exposure_ratio = _require_finite_scalar(short_exposure_ratio, "空頭曝險比例")

    total_turnover = turnover_per_bar = None
    rebalance_count = None
    if turn is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            total_turnover = _require_finite_scalar(float(turn.sum()), "總換手")
            turnover_per_bar = _require_finite_scalar(float(turn.mean()), "每期換手")
        rebalance_count = int(np.count_nonzero(turn > 1e-12))

    unavailable: dict[str, str] = {}
    if sharpe is None:
        unavailable["sharpe"] = "報酬波動為零，無法計算夏普比率"
    if sortino is None:
        unavailable["sortino"] = "下行報酬波動為零，無法計算索提諾比率"
    if profit_factor is None:
        unavailable["profit_factor"] = "沒有足夠的虧損報酬，無法計算獲利因子"

    return {
        "bars": int(len(pnl)),
        "total_log_return": total_log,
        "total_return": total_return,
        "annual_log_return": annual_log,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "profit_factor": profit_factor,
        "exposure_ratio": exposure_ratio,
        "mean_absolute_position": mean_absolute_position,
        "long_exposure_ratio": long_exposure_ratio,
        "short_exposure_ratio": short_exposure_ratio,
        "total_turnover": total_turnover,
        "turnover_per_bar": turnover_per_bar,
        "rebalance_count": rebalance_count,
        "unavailable_reasons": unavailable,
    }


def cost_stress(
    position: np.ndarray,
    market_return: np.ndarray,
    base_cost: float,
    multipliers: Iterable[float],
    periods_per_year: int,
) -> dict[str, dict[str, Any]]:
    """在固定成本倍數下重算績效，供成本壓力測試使用。"""
    base = _as_non_negative_finite_scalar(base_cost, "基準單邊成本")
    scenarios: dict[str, dict[str, Any]] = {}
    for multiplier in multipliers:
        factor = _as_non_negative_finite_scalar(multiplier, "成本倍數")
        scenario_cost = _require_finite_scalar(base * factor, "情境單邊成本")
        net_returns, turnover = build_net_returns(position, market_return, scenario_cost)
        metrics = performance_metrics(
            net_returns, position, periods_per_year, turnover=turnover
        )
        metrics["one_way_cost"] = scenario_cost
        scenarios[f"{factor:g}x"] = metrics
    return scenarios


def concentration_analysis(block_returns: Iterable[float]) -> dict[str, Any]:
    """檢查整體報酬是否過度依賴少數時間區塊。"""
    values = _as_finite_vector(np.asarray(list(block_returns), dtype=np.float64), "區塊報酬")
    if len(values) < 2:
        return {
            "severe": True,
            "reason": "時間區塊不足",
            "best_block_share": None,
            "top_three_share": None,
            "return_without_best_block": None,
        }

    total_positive = float(values[values > 0].sum())
    ordered = np.sort(values)[::-1]
    best_share = float(ordered[0] / total_positive) if total_positive > 1e-12 else None
    top_three_share = (
        float(ordered[:3].clip(min=0).sum() / total_positive)
        if total_positive > 1e-12
        else None
    )
    return_without_best_block = float(values.sum() - ordered[0])
    severe = (
        best_share is None
        or best_share > 0.50
        or return_without_best_block <= 0
    )
    return {
        "severe": severe,
        "reason": "報酬過度集中於少數區塊" if severe else "報酬未顯示嚴重集中",
        "best_block_share": best_share,
        "top_three_share": top_three_share,
        "return_without_best_block": return_without_best_block,
    }


def _summarize_blocks(
    blocks: Iterable[np.ndarray], pnl: np.ndarray, timestamps: pd.DatetimeIndex
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    returns: list[float] = []
    for indices in blocks:
        if len(indices) == 0:
            continue
        block_return = _require_finite_scalar(float(pnl[indices].sum()), "區塊報酬")
        returns.append(block_return)
        details.append(
            {
                "start_time": timestamps[indices[0]].isoformat(),
                "end_time": timestamps[indices[-1]].isoformat(),
                "bars": int(len(indices)),
                "return": block_return,
            }
        )
    values = np.asarray(returns, dtype=np.float64)
    return {
        "blocks": details,
        "returns": returns,
        "median_return": float(np.median(values)) if len(values) else None,
        "positive_ratio": float(np.mean(values > 0)) if len(values) else None,
    }


def temporal_blocks(
    net_returns: np.ndarray,
    times: np.ndarray,
    periods_per_year: int,
    equal_blocks: int = 8,
) -> dict[str, Any]:
    """以自然年度與等長區塊檢查策略報酬的時間穩定性。"""
    pnl = _as_finite_vector(net_returns, "淨報酬")
    _as_positive_finite_scalar(periods_per_year, "每年期數")
    block_count = _as_positive_integer(equal_blocks, "等長區塊數")
    timestamps = _time_index(times, len(pnl))

    years = timestamps.year.to_numpy()
    natural_indices = [np.flatnonzero(years == year) for year in np.unique(years)]
    natural = _summarize_blocks(natural_indices, pnl, timestamps)
    for detail, year in zip(natural["blocks"], np.unique(years), strict=True):
        detail["year"] = int(year)

    equal = _summarize_blocks(
        np.array_split(np.arange(len(pnl)), block_count), pnl, timestamps
    )
    concentration = concentration_analysis(equal["returns"])
    return {
        "natural_year": natural,
        "equal": equal,
        "concentration": concentration,
    }


def market_regression(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
) -> dict[str, Any]:
    """以 OLS 分離市場 beta 與策略 alpha。"""
    strategy = np.asarray(strategy_returns, dtype=np.float64).reshape(-1)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64).reshape(-1)
    periods = _as_positive_finite_scalar(periods_per_year, "每年期數")
    if len(strategy) != len(benchmark):
        raise ValueError("策略與基準報酬長度必須一致")

    finite = np.isfinite(strategy) & np.isfinite(benchmark)
    strategy = strategy[finite]
    benchmark = benchmark[finite]
    observations = int(len(strategy))
    if observations < 30:
        return _unavailable_regression("可用的策略與基準報酬配對不足 30 筆", observations)
    if (
        float(np.std(strategy, ddof=0)) <= 1e-12
        or float(np.std(benchmark, ddof=0)) <= 1e-12
    ):
        return _unavailable_regression(
            "策略或基準報酬缺乏變異，無法估計回歸指標", observations
        )

    design = np.column_stack([np.ones(observations), benchmark])
    coefficients, _, _, _ = np.linalg.lstsq(design, strategy, rcond=None)
    alpha = _require_finite_scalar(float(coefficients[0]), "回歸截距")
    beta = _require_finite_scalar(float(coefficients[1]), "市場 beta")
    residuals = strategy - design @ coefficients
    residual_mean = _require_finite_scalar(float(residuals.mean()), "殘差平均值")
    residual_std = _require_finite_scalar(float(residuals.std(ddof=0)), "殘差標準差")
    if residual_std <= 1e-12:
        return _unavailable_regression(
            "殘差標準差為零，無法計算回歸指標", observations
        )
    residual_sharpe = _require_finite_scalar(
        residual_mean / residual_std * math.sqrt(periods), "殘差 Sharpe"
    )
    correlation = _require_finite_scalar(
        float(np.corrcoef(strategy, benchmark)[0, 1]), "策略與基準相關性"
    )
    return {
        "observations": observations,
        "alpha": alpha,
        "annual_alpha": _require_finite_scalar(alpha * periods, "年化 alpha"),
        "beta": beta,
        "residual_sharpe": residual_sharpe,
        "correlation": correlation,
        "reason": None,
    }


def pseudo_walk_forward(
    net_returns: np.ndarray,
    times: np.ndarray,
    periods_per_year: int,
    splits: int = 5,
    embargo_bars: int = 24,
) -> dict[str, Any]:
    """以 embargo 分隔時間區塊；結果不應視為真正樣本外驗證。"""
    pnl = _as_finite_vector(net_returns, "淨報酬")
    timestamps = _time_index(times, len(pnl))
    periods = _as_positive_finite_scalar(periods_per_year, "每年期數")
    split_count = _as_positive_integer(splits, "切分數")
    embargo = _as_positive_integer(embargo_bars + 1, "隔離期") - 1

    folds: list[dict[str, Any]] = []
    for fold_number, indices in enumerate(np.array_split(np.arange(len(pnl)), split_count)):
        retained = indices if fold_number == 0 else indices[embargo:]
        start_index = int(retained[0]) if len(retained) else None
        end_index = int(retained[-1]) + 1 if len(retained) else None
        fold: dict[str, Any] = {
            "fold": fold_number + 1,
            "start_index": start_index,
            "end_index": end_index,
            "embargo_bars": 0 if fold_number == 0 else embargo,
            "start_time": timestamps[retained[0]].isoformat() if len(retained) else None,
            "end_time": timestamps[retained[-1]].isoformat() if len(retained) else None,
        }
        if len(retained):
            fold["metrics"] = performance_metrics(
                pnl[retained], np.ones(len(retained)), periods
            )
            fold["reason"] = None
        else:
            fold["metrics"] = None
            fold["reason"] = "隔離期後沒有可評估的資料"
        folds.append(fold)

    return {
        "folds": folds,
        "is_true_out_of_sample": False,
        "limitation": "歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。",
    }
