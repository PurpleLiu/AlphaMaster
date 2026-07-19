"""Render local, reproducible reports for an AlphaMaster strategy judgment."""
from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


MAX_REPORT_FILENAME_LENGTH = 120
_MAX_SYMBOL_FILENAME_PART_LENGTH = 48
_MAX_TIMEFRAME_FILENAME_PART_LENGTH = 32


def to_json_safe(value: Any) -> Any:
    """Recursively convert a value into strict JSON-compatible data.

    JSON has no representation for NaN or infinities.  They are intentionally
    rendered as ``null`` so an unavailable metric cannot be mistaken for zero.
    """
    if isinstance(value, np.timedelta64):
        return None if np.isnat(value) else str(value)
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else str(value)
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        kind = value.dtype.kind
        if kind in {"i", "u"}:
            return int(value)
        if kind == "b":
            return bool(value)
        if kind == "f":
            if not bool(np.isfinite(value)):
                return None
            converted = float(value)
            return converted if math.isfinite(converted) else None
        if kind == "c":
            real = value.real
            imaginary = value.imag
            if not bool(np.isfinite(real)) or not bool(np.isfinite(imaginary)):
                return None
            # JSON has no complex-number type.  Consume both components before
            # returning null so extended-precision complex scalars cannot recur.
            return None
        if kind == "S":
            return to_json_safe(bytes(value))
        if kind == "U":
            return str(value)
        if kind == "V":
            return to_json_safe(value.tolist())
        item = value.item()
        if item is value:
            return str(value)
        return to_json_safe(item)
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
        return seconds if math.isfinite(seconds) else None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, complex):
        return None
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any) -> str:
    safe = to_json_safe(value)
    if safe is None:
        return "無法取得"
    if isinstance(safe, float):
        return f"{safe:.6g}"
    return str(safe)


def _reason_value(value: Any, *, data_available: bool) -> str:
    """Render diagnostic reasons without treating missing metadata as failure."""
    if value is None and data_available:
        return "—"
    return _value(value)


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
        data_available = any(value is not None for value in metrics.values())
        rows.append(
            f"| {_value(item.get('fold'))} | {_value(item.get('start_time'))} | "
            f"{_value(item.get('end_time'))} | {_value(metrics.get('sharpe'))} | "
            f"{_reason_value(item.get('reason'), data_available=data_available)} |"
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
                    (
                        "回歸說明",
                        _reason_value(
                            regression.get("reason"),
                            data_available=any(
                                regression.get(field) is not None
                                for field in (
                                    "beta",
                                    "annual_alpha",
                                    "residual_sharpe",
                                    "correlation",
                                )
                            ),
                        ),
                    ),
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


def _filename_part(value: Any, fallback: str, max_length: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return (cleaned[:max_length].strip("._-") or fallback)[:max_length]


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


def _write_temp_text(directory: Path, final_name: str, content: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_name}.", suffix=".tmp", dir=directory, text=True
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sync_directory(directory: Path) -> None:
    """Persist directory entries when the current platform supports it."""
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.bak")


def _validate_final_target(path: Path, label: str) -> None:
    """Reject unsafe existing report targets before publication begins."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return

    if path.is_symlink():
        raise ValueError(f"報告 {label} 目標不可是符號連結：{path}")
    if stat.S_ISREG(metadata.st_mode):
        return
    if stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"報告 {label} 目標不可是目錄：{path}")
    raise ValueError(f"報告 {label} 目標必須是一般檔案：{path}")


def _remove(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def write_reports(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """Write strict JSON and Markdown reports under a reproducible safe filename."""
    safe = to_json_safe(result)
    source = _mapping(safe.get("source"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    symbol = _filename_part(
        source.get("symbol"), "UNKNOWN", _MAX_SYMBOL_FILENAME_PART_LENGTH
    )
    timeframe = _filename_part(
        source.get("timeframe"), "UNKNOWN", _MAX_TIMEFRAME_FILENAME_PART_LENGTH
    )
    digest = _filename_part(source.get("strategy_sha256"), "nohash", 8)
    timestamp = _utc_file_timestamp(safe.get("created_at_utc") or safe.get("created_at"))
    stem = f"{symbol}_{timeframe}_{digest}_{timestamp}"
    json_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"

    if max(len(json_path.name), len(markdown_path.name)) > MAX_REPORT_FILENAME_LENGTH:
        raise ValueError("審判報告檔名超過安全長度限制")

    _validate_final_target(json_path, "JSON")
    _validate_final_target(markdown_path, "Markdown")

    json_content = json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown_content = render_markdown(safe)
    json_temp: Path | None = None
    markdown_temp: Path | None = None
    json_backup: Path | None = None
    markdown_backup: Path | None = None
    json_published = False
    markdown_published = False

    try:
        json_temp = _write_temp_text(output_path, json_path.name, json_content)
        markdown_temp = _write_temp_text(output_path, markdown_path.name, markdown_content)

        if json_path.exists():
            json_backup = _backup_path(json_path)
            os.replace(json_path, json_backup)
        if markdown_path.exists():
            markdown_backup = _backup_path(markdown_path)
            os.replace(markdown_path, markdown_backup)

        os.replace(json_temp, json_path)
        json_temp = None
        json_published = True
        os.replace(markdown_temp, markdown_path)
        markdown_temp = None
        markdown_published = True
        _sync_directory(output_path)
    except BaseException:
        _remove(json_temp)
        _remove(markdown_temp)
        if json_published:
            _remove(json_path)
        if markdown_published:
            _remove(markdown_path)
        if json_backup is not None:
            os.replace(json_backup, json_path)
        if markdown_backup is not None:
            os.replace(markdown_backup, markdown_path)
        raise
    else:
        _remove(json_backup)
        _remove(markdown_backup)
    return json_path, markdown_path
