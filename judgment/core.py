from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name}含有非有限值，無法進行計算")
    return vector


def _validate_periods_per_year(periods_per_year: int) -> int:
    if isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("每年期數必須是大於零的整數")
    return periods_per_year


def build_net_returns(
    position: np.ndarray,
    market_return: np.ndarray,
    one_way_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """依每次部位變動扣除單邊交易成本，回傳淨報酬與換手。"""
    pos = _as_finite_vector(position, "部位")
    ret = _as_finite_vector(market_return, "市場報酬")
    cost = float(one_way_cost)
    if pos.shape != ret.shape:
        raise ValueError("部位與市場報酬長度必須一致")
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("單邊成本必須是大於或等於零的有限數值")

    previous = np.concatenate(([0.0], pos[:-1]))
    turnover = np.abs(pos - previous)
    return pos * ret - turnover * cost, turnover


def performance_metrics(
    net_returns: np.ndarray,
    position: np.ndarray,
    periods_per_year: int,
    turnover: np.ndarray | None = None,
) -> dict[str, Any]:
    """計算淨對數報酬的績效指標，並明確標示不可用的指標。"""
    pnl = _as_finite_vector(net_returns, "淨報酬")
    pos = _as_finite_vector(position, "部位")
    periods = _validate_periods_per_year(periods_per_year)
    if len(pnl) == 0:
        raise ValueError("淨報酬與部位不可為空")
    if len(pnl) != len(pos):
        raise ValueError("淨報酬與部位長度必須一致")

    turn = None
    if turnover is not None:
        turn = _as_finite_vector(turnover, "換手")
        if len(turn) != len(pnl):
            raise ValueError("換手與淨報酬長度必須一致")

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=0))
    downside = pnl[pnl < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else None
    sharpe = mean / std * math.sqrt(periods) if std > 1e-12 else None
    sortino = (
        mean / downside_std * math.sqrt(periods)
        if downside_std is not None and downside_std > 1e-12
        else None
    )

    curve = np.cumsum(pnl)
    drawdown = curve - np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:]
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    profit_factor = gains / losses if losses > 1e-12 else None
    total_log = float(pnl.sum())
    years = len(pnl) / float(periods)

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
        "total_return": float(np.expm1(total_log)),
        "annual_log_return": total_log / years,
        "annual_return": float(np.expm1(total_log / years)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "profit_factor": profit_factor,
        "exposure_ratio": float(np.mean(np.abs(pos) > 1e-12)),
        "mean_absolute_position": float(np.mean(np.abs(pos))),
        "long_exposure_ratio": float(np.mean(pos > 1e-12)),
        "short_exposure_ratio": float(np.mean(pos < -1e-12)),
        "total_turnover": float(turn.sum()) if turn is not None else None,
        "turnover_per_bar": float(turn.mean()) if turn is not None else None,
        "rebalance_count": int(np.count_nonzero(turn > 1e-12)) if turn is not None else None,
        "unavailable_reasons": unavailable,
    }


def cost_stress(
    position: np.ndarray,
    market_return: np.ndarray,
    base_cost: float,
    multipliers: Iterable[float],
    periods_per_year: int,
) -> dict[str, dict[str, Any]]:
    """以多個成本倍數重算績效，供交易成本壓力測試使用。"""
    base = float(base_cost)
    if not math.isfinite(base) or base < 0:
        raise ValueError("基準單邊成本必須是大於或等於零的有限數值")

    scenarios: dict[str, dict[str, Any]] = {}
    for multiplier in multipliers:
        factor = float(multiplier)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("成本倍數必須是大於或等於零的有限數值")
        scenario_cost = base * factor
        net_returns, turnover = build_net_returns(position, market_return, scenario_cost)
        metrics = performance_metrics(
            net_returns, position, periods_per_year, turnover=turnover
        )
        metrics["one_way_cost"] = scenario_cost
        scenarios[f"{factor:g}x"] = metrics
    return scenarios
