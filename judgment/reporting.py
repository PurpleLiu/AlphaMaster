"""Render local, reproducible reports for an AlphaMaster strategy judgment."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """Recursively convert a value into strict JSON-compatible data.

    JSON has no representation for NaN or infinities.  They are intentionally
    rendered as ``null`` so an unavailable metric cannot be mistaken for zero.
    """
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any) -> str:
    safe = to_json_safe(value)
    if safe is None:
        return "無法取得"
    if isinstance(safe, float):
        return f"{safe:.6g}"
    return str(safe)


def _table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| 項目 | 結果 |", "| --- | ---: |"]
    lines.extend(f"| {name} | {_value(value)} |" for name, value in rows)
    return "\n".join(lines)


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}"


def _rules_markdown(verdict: Mapping[str, Any]) -> str:
    rows = ["| 規則 | 目前值 | 門檻 | 結果 | 說明 |", "| --- | ---: | ---: | --- | --- |"]
    for rule in verdict.get("rules", []):
        item = _mapping(rule)
        passed = item.get("passed")
        outcome = "通過" if passed is True else "未通過" if passed is False else "需人工審查"
        rows.append(
            "| {name} | {value} | {operator} {threshold} | {outcome} | {explanation} |".format(
                name=_value(item.get("name")),
                value=_value(item.get("value")),
                operator=_value(item.get("operator")),
                threshold=_value(item.get("threshold")),
                outcome=outcome,
                explanation=_value(item.get("explanation")),
            )
        )
    return "\n".join(rows)


def _stress_markdown(stress: Mapping[str, Any]) -> str:
    rows = ["| 成本情境 | 單邊成本 | 總對數報酬 | 夏普比率 | 獲利因子 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in ("1x", "2x", "3x", "5x"):
        metrics = _mapping(stress.get(label))
        rows.append(
            f"| {label} | {_value(metrics.get('one_way_cost'))} | "
            f"{_value(metrics.get('total_log_return'))} | {_value(metrics.get('sharpe'))} | "
            f"{_value(metrics.get('profit_factor'))} |"
        )
    return "\n".join(rows)


def _walk_forward_markdown(walk_forward: Mapping[str, Any]) -> str:
    rows = ["| Fold | 起始時間 | 結束時間 | 夏普比率 | 說明 |", "| ---: | --- | --- | ---: | --- |"]
    for fold in walk_forward.get("folds", []):
        item = _mapping(fold)
        metrics = _mapping(item.get("metrics"))
        rows.append(
            f"| {_value(item.get('fold'))} | {_value(item.get('start_time'))} | "
            f"{_value(item.get('end_time'))} | {_value(metrics.get('sharpe'))} | "
            f"{_value(item.get('reason'))} |"
        )
    return "\n".join(rows)


def render_markdown(result: dict) -> str:
    """Render a Taiwan Traditional Chinese review report in a stable section order."""
    safe = to_json_safe(result)
    source = _mapping(safe.get("source"))
    verdict = _mapping(safe.get("verdict"))
    full_sample = _mapping(safe.get("full_sample"))
    temporal = _mapping(safe.get("temporal_blocks") or safe.get("temporal"))
    equal = _mapping(temporal.get("equal"))
    concentration = _mapping(safe.get("concentration") or temporal.get("concentration"))
    regression = _mapping(safe.get("regression") or safe.get("market_regression"))
    walk_forward = _mapping(safe.get("walk_forward"))
    limitations = _mapping(safe.get("limitations"))
    status = _value(verdict.get("status", "REVIEW"))

    sections = [
        "# AlphaMaster 策略審判報告",
        _section("判決：" + status, "此結果是成本、時間穩定性與市場曝險檢查後的策略審判結果。"),
        _section(
            "策略來源與資料範圍",
            _table(
                [
                    ("策略 SHA-256", source.get("strategy_sha256")),
                    ("策略檔案", source.get("strategy_path")),
                    ("資料檔案", source.get("data_path")),
                    ("品種", source.get("symbol")),
                    ("週期", source.get("timeframe")),
                    ("資料筆數（bars）", source.get("bars")),
                    ("資料開始", source.get("start_time")),
                    ("資料結束", source.get("end_time")),
                ]
            ),
        ),
        _section(
            "完整樣本績效",
            _table(
                [
                    ("總對數報酬", full_sample.get("total_log_return")),
                    ("總報酬", full_sample.get("total_return")),
                    ("年化報酬", full_sample.get("annual_return")),
                    ("夏普比率", full_sample.get("sharpe")),
                    ("索提諾比率", full_sample.get("sortino")),
                    ("獲利因子", full_sample.get("profit_factor")),
                    ("最大回撤", full_sample.get("max_drawdown")),
                    ("總換手", full_sample.get("total_turnover")),
                ]
            ),
        ),
        _section("判決規則", _rules_markdown(verdict)),
        _section("成本壓力測試", _stress_markdown(_mapping(safe.get("cost_stress")))),
        _section(
            "時間穩定性與集中度",
            _table(
                [
                    ("等分區塊報酬中位數", equal.get("median_return")),
                    ("正報酬區塊比例", equal.get("positive_ratio")),
                    ("是否嚴重集中", concentration.get("severe")),
                    ("集中度說明", concentration.get("reason")),
                    ("最佳區塊報酬占比", concentration.get("best_block_share")),
                ]
            ),
        ),
        _section(
            "BTC Beta／Alpha",
            _table(
                [
                    ("BTC Beta", regression.get("beta")),
                    ("年度 Alpha", regression.get("annual_alpha")),
                    ("殘差夏普比率", regression.get("residual_sharpe")),
                    ("相關係數", regression.get("correlation")),
                    ("回歸說明", regression.get("reason")),
                ]
            ),
        ),
        _section("Pseudo Walk-Forward", _walk_forward_markdown(walk_forward)),
        _section(
            "重要限制與非真正樣本外警告",
            "\n".join(
                [
                    "**警告：這不是真正的樣本外驗證。**",
                    _value(limitations.get("pseudo_walk_forward")),
                    _value(limitations.get("true_out_of_sample")),
                    "策略曾以歷史資料搜尋與挑選；真正樣本外驗證須使用未參與搜尋的未來資料，或先進行 dry-run。",
                ]
            ),
        ),
    ]
    return "\n\n".join(sections) + "\n"


def _filename_part(value: Any, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or fallback


def _utc_file_timestamp(value: Any) -> str:
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_reports(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """Write strict JSON and Markdown reports under a reproducible safe filename."""
    safe = to_json_safe(result)
    source = _mapping(safe.get("source"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    symbol = _filename_part(source.get("symbol"), "UNKNOWN")
    timeframe = _filename_part(source.get("timeframe"), "UNKNOWN")
    digest = _filename_part(source.get("strategy_sha256"), "nohash")[:8]
    timestamp = _utc_file_timestamp(safe.get("created_at_utc") or safe.get("created_at"))
    stem = f"{symbol}_{timeframe}_{digest}_{timestamp}"
    json_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"

    json_path.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(safe), encoding="utf-8")
    return json_path, markdown_path
