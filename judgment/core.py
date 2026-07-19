from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


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
