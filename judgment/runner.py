"""Reproduce an AlphaMaster strategy and assemble an independent judgment report."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.backtest import estimate_periods_per_year
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.signal import compute_target_positions_stateless

from .core import (
    build_net_returns,
    classify_judgment,
    cost_stress,
    market_regression,
    performance_metrics,
    pseudo_walk_forward,
    temporal_blocks,
)


def _formula_from_payload(payload: Any) -> list[int]:
    """Extract the one supported, non-empty integer token formula."""
    formula = payload.get("formula") if isinstance(payload, Mapping) else payload
    if not isinstance(formula, list) or not formula:
        raise ValueError("策略快照必須包含非空的整數 formula")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in formula):
        raise ValueError("策略快照的 formula 必須全部是整數")
    invalid_tokens = [
        token for token in formula if token < 0 or token >= FORMULA_VOCAB.size
    ]
    if invalid_tokens:
        raise ValueError(
            "策略快照的 formula 含有超出目前詞彙表範圍的 token "
            f"（合法範圍：0 至 {FORMULA_VOCAB.size - 1}）：{invalid_tokens}"
        )
    return formula


def load_strategy_snapshot(path: str | Path) -> dict[str, Any]:
    """Read one immutable strategy-byte snapshot and derive its provenance."""
    strategy_path = Path(path).resolve()
    raw_bytes = strategy_path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("策略快照必須是 UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError("策略快照不是有效的 JSON") from error

    formula = _formula_from_payload(payload)
    return {
        "resolved_path": str(strategy_path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "payload": payload,
        "formula": formula,
    }


def _utc_timestamp(seconds: float) -> str:
    return pd.Timestamp(seconds, unit="s", tz="UTC").isoformat()


def _unavailable_reasons(
    full_sample: dict[str, Any],
    stress: dict[str, dict[str, Any]],
    regression: dict[str, Any],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    """Keep every unavailable explanation emitted by the core diagnostics."""
    reasons: dict[str, Any] = {
        "full_sample": full_sample.get("unavailable_reasons", {}),
        "cost_stress": {
            label: metrics.get("unavailable_reasons", {})
            for label, metrics in stress.items()
        },
        "regression": regression.get("reason"),
        "walk_forward": {
            str(fold["fold"]): fold["reason"]
            for fold in walk_forward["folds"]
            if fold.get("reason")
        },
    }
    return reasons


def run_judgment(
    strategy_path: str | Path,
    data_path: str | Path,
    base_cost: float = 0.0006,
    stress_multipliers: tuple[float, ...] = (1, 2, 3, 5),
    equal_blocks: int = 8,
    walk_forward_splits: int = 5,
    embargo_bars: int = 24,
) -> dict[str, Any]:
    """Reconstruct a saved factor's positions and apply all judgment diagnostics.

    This intentionally uses AlphaMaster's production feature, VM, position, and
    market-return machinery, while every performance and verdict calculation is
    delegated to the independent ``judgment.core`` module.
    """
    snapshot = load_strategy_snapshot(strategy_path)
    manager = ParquetDataManager(data_path)
    manager.load()

    raw = manager.raw_dict
    features = manager.feat_tensor
    factor = StackVM().execute(snapshot["formula"], features)
    if factor is None:
        raise ValueError(f"StackVM 無法執行策略公式: {snapshot['formula']}")

    position = compute_target_positions_stateless(factor)[0].detach().cpu().numpy()
    market_return = manager.target_ret[0].detach().cpu().numpy()
    times = raw["time"][0].detach().cpu().numpy()
    periods_per_year = estimate_periods_per_year(raw["time"])

    net_returns, turnover = build_net_returns(position, market_return, base_cost)
    full_sample = performance_metrics(
        net_returns, position, periods_per_year, turnover=turnover
    )
    stress = cost_stress(
        position,
        market_return,
        base_cost,
        stress_multipliers,
        periods_per_year,
    )
    blocks = temporal_blocks(
        net_returns,
        times,
        periods_per_year,
        equal_blocks=equal_blocks,
    )
    concentration = blocks["concentration"]
    regression = market_regression(net_returns, market_return, periods_per_year)
    walk_forward = pseudo_walk_forward(
        net_returns,
        times,
        periods_per_year,
        splits=walk_forward_splits,
        embargo_bars=embargo_bars,
    )
    verdict = classify_judgment(
        full_sample,
        stress,
        blocks,
        regression,
        concentration,
    )

    data_file = Path(data_path).resolve()
    first_time = float(np.asarray(times).reshape(-1)[0])
    last_time = float(np.asarray(times).reshape(-1)[-1])
    pseudo_oos_limitation = walk_forward["limitation"]
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "strategy_path": snapshot["resolved_path"],
            "strategy_sha256": snapshot["sha256"],
            "strategy_payload": snapshot["payload"],
            "formula": snapshot["formula"],
            "data_path": str(data_file),
            "symbol": manager.symbol,
            "timeframe": manager.timeframe,
            "bars": int(len(position)),
            "start_time": _utc_timestamp(first_time),
            "end_time": _utc_timestamp(last_time),
        },
        "assumptions": {
            "base_one_way_cost": base_cost,
            "stress_multipliers": list(stress_multipliers),
            "equal_blocks": equal_blocks,
            "walk_forward_splits": walk_forward_splits,
            "embargo_bars": embargo_bars,
            "periods_per_year": periods_per_year,
        },
        "full_sample": full_sample,
        "cost_stress": stress,
        "temporal_blocks": blocks,
        "concentration": concentration,
        "regression": regression,
        "walk_forward": walk_forward,
        "verdict": verdict,
        "verdict_rules": verdict["rules"],
        "unavailable_reasons": _unavailable_reasons(
            full_sample, stress, regression, walk_forward
        ),
        "limitations": {
            "is_true_out_of_sample": walk_forward["is_true_out_of_sample"],
            "pseudo_walk_forward": pseudo_oos_limitation,
            "true_out_of_sample": (
                "歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。"
                "真正樣本外驗證需要未參與搜尋的未來資料或 dry-run。"
            ),
        },
    }
