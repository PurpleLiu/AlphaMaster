from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import judgment.reporting as reporting
from judgment.reporting import render_markdown, to_json_safe, write_reports


def _result() -> dict:
    return {
        "created_at_utc": "2026-07-19T12:34:56+00:00",
        "source": {
            "symbol": "BTC/USDT?*",
            "timeframe": "H1 test",
            "strategy_sha256": "abcdef1234567890",
            "strategy_path": "strategies/best_BTCUSDT.json",
            "data_path": "data/BTCUSDT_H1.parquet",
            "bars": 240,
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-11T00:00:00+00:00",
        },
        "assumptions": {"base_one_way_cost": 0.0006, "periods_per_year": 8760},
        "verdict": {
            "status": "REVIEW",
            "rules": [
                {
                    "name": "sharpe",
                    "value": 0.5,
                    "threshold": 0.75,
                    "operator": ">=",
                    "passed": False,
                    "explanation": "夏普比率需 >= 0.75；目前值為 0.50。",
                }
            ],
        },
        "full_sample": {"sharpe": 0.5, "profit_factor": 1.1},
        "cost_stress": {
            "1x": {"total_log_return": np.float64(0.1)},
            "2x": {"total_log_return": np.nan},
            "3x": {"total_log_return": np.inf},
            "5x": {"total_log_return": -np.inf},
        },
        "temporal_blocks": {
            "equal": {"median_return": 0.01, "positive_ratio": 0.75},
        },
        "concentration": {"severe": False, "reason": "報酬未顯示嚴重集中"},
        "regression": {"beta": 0.8, "annual_alpha": 0.12, "residual_sharpe": 0.6},
        "walk_forward": {"folds": [{"fold": 1, "metrics": {"sharpe": 0.2}}]},
        "limitations": {
            "is_true_out_of_sample": False,
            "pseudo_walk_forward": "歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。",
            "true_out_of_sample": "真正樣本外驗證需要未參與搜尋的未來資料或 dry-run。",
        },
    }


def test_to_json_safe_recursively_converts_numpy_and_nonfinite_to_null() -> None:
    result = to_json_safe(
        {
            "scalar": np.float64(1.2),
            "array": np.array([np.int64(3), np.nan, np.inf]),
            "nested": [np.float32("-inf"), {"ok": True}],
        }
    )

    assert result == {
        "scalar": 1.2,
        "array": [3, None, None],
        "nested": [None, {"ok": True}],
    }
    assert json.loads(json.dumps(result)) == result


def test_to_json_safe_converts_datetime_like_numpy_scalars_to_text() -> None:
    result = to_json_safe(
        {
            "python": datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc),
            "numpy": np.datetime64("2026-07-19T12:00:00"),
        }
    )

    assert result == {
        "python": "2026-07-19T12:00:00+00:00",
        "numpy": "2026-07-19T12:00:00",
    }
    assert json.loads(json.dumps(result)) == result


def test_to_json_safe_converts_nonstandard_numpy_and_complex_values_to_strict_json() -> None:
    result = to_json_safe(
        {
            "duration": np.timedelta64(3, "h"),
            "missing_duration": np.timedelta64("NaT", "ns"),
            "bytes": np.bytes_(b"alpha"),
            "invalid_bytes": b"\xff",
            "complex": 2 + 3j,
            "complex_array": np.array([1 + 2j, 3 - 4j]),
        }
    )

    assert result == {
        "duration": "3 hours",
        "missing_duration": None,
        "bytes": "alpha",
        "invalid_bytes": "�",
        "complex": None,
        "complex_array": [None, None],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_to_json_safe_converts_extended_precision_numpy_scalars_without_recursion() -> None:
    result = to_json_safe(
        {
            "longdouble": np.longdouble("1.25"),
            "longdouble_nan": np.longdouble("nan"),
            "longdouble_inf": np.longdouble("inf"),
            "clongdouble": np.clongdouble("1.25+2.5j"),
            "clongdouble_nan": np.clongdouble(complex(float("nan"), 0.0)),
        }
    )

    assert result == {
        "longdouble": 1.25,
        "longdouble_nan": None,
        "longdouble_inf": None,
        "clongdouble": None,
        "clongdouble_nan": None,
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_to_json_safe_converts_extended_precision_numpy_arrays_without_recursion() -> None:
    result = to_json_safe(
        {
            "longdouble": np.array([np.longdouble("2.5"), np.longdouble("nan")]),
            "clongdouble": np.array(
                [np.clongdouble("3+4j"), np.clongdouble(complex(0.0, float("inf")))],
                dtype=np.clongdouble,
            ),
        }
    )

    assert result == {
        "longdouble": [2.5, None],
        "clongdouble": [None, None],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_markdown_lists_required_diagnostics_and_prominent_oos_warning() -> None:
    text = render_markdown(_result())

    headings = [
        "# AlphaMaster 策略審判報告",
        "## 判決：REVIEW",
        "## 策略來源與資料範圍",
        "## 完整樣本績效",
        "## 判決規則",
        "## 成本壓力測試",
        "## 時間穩定性與集中度",
        "## BTC Beta／Alpha",
        "## Pseudo Walk-Forward",
        "## 重要限制與非真正樣本外警告",
    ]
    positions = [text.index(heading) for heading in headings]

    assert positions == sorted(positions)
    assert "BTC/USDT?*" in text
    assert "abcdef1234567890" in text
    assert "1x" in text and "2x" in text and "3x" in text and "5x" in text
    assert "夏普比率需 >= 0.75" in text
    assert "**警告：這不是真正的樣本外驗證。**" in text
    assert "歷史資料曾參與策略搜尋" in text


def test_markdown_distinguishes_missing_reason_from_unavailable_diagnostics() -> None:
    result = _result()
    result["regression"] = {
        "beta": 0.8,
        "annual_alpha": 0.12,
        "residual_sharpe": 0.6,
        "correlation": 0.4,
        "reason": None,
    }
    result["walk_forward"] = {
        "folds": [
            {
                "fold": 1,
                "start_time": "s1",
                "end_time": "e1",
                "metrics": {"sharpe": 0.2},
                "reason": None,
            },
            {
                "fold": 2,
                "start_time": "s2",
                "end_time": "e2",
                "metrics": {"sharpe": 0.3},
            },
            {
                "fold": 3,
                "start_time": "s3",
                "end_time": "e3",
                "metrics": {"sharpe": 0.4},
                "reason": "資料不足",
            },
            {
                "fold": 4,
                "start_time": "s4",
                "end_time": "e4",
                "metrics": {"sharpe": 0.5},
                "reason": "無法取得",
            },
        ]
    }

    text = render_markdown(result)

    assert "BTC Beta" in text
    assert "| 回歸說明 | — |" in text
    assert "| 1 | s1 | e1 | 0.2 | — |" in text
    assert "| 2 | s2 | e2 | 0.3 | — |" in text
    assert "| 3 | s3 | e3 | 0.4 | 資料不足 |" in text
    assert "| 4 | s4 | e4 | 0.5 | 無法取得 |" in text


def test_reports_recursively_redact_sensitive_strategy_metadata(tmp_path: Path) -> None:
    result = _result()
    secret_values = ["report-api-secret", "report-nested-token", "report-chat-id"]
    result["source"]["strategy_metadata"] = {
        "score": 1.5,
        "api_key": secret_values[0],
        "nested": {
            "token": secret_values[1],
            "chat_id": secret_values[2],
        },
    }

    rendered = render_markdown(result)
    json_path, markdown_path = write_reports(result, tmp_path)
    for secret in secret_values:
        assert secret not in rendered
        assert secret not in json_path.read_text(encoding="utf-8")
        assert secret not in markdown_path.read_text(encoding="utf-8")

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = saved["source"]["strategy_metadata"]
    assert metadata["api_key"] == "***已遮蔽***"
    assert metadata["nested"]["token"] == "***已遮蔽***"
    assert metadata["nested"]["chat_id"] == "***已遮蔽***"


def test_write_reports_uses_safe_deterministic_filename_and_valid_utf8_json(tmp_path: Path) -> None:
    result = _result()

    json_path, markdown_path = write_reports(result, tmp_path)

    assert json_path.parent == tmp_path
    assert markdown_path.parent == tmp_path
    assert json_path.stem == markdown_path.stem
    assert json_path.name.startswith("BTC_USDT_H1_test_abcdef12_20260719_123456")
    assert json_path.suffix == ".json"
    assert markdown_path.suffix == ".md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["cost_stress"]["2x"]["total_log_return"] is None
    assert saved["cost_stress"]["3x"]["total_log_return"] is None
    assert markdown_path.read_text(encoding="utf-8").startswith("# AlphaMaster 策略審判報告")


def test_write_reports_bounds_long_filename_parts_without_losing_hash_or_timestamp(tmp_path: Path) -> None:
    result = _result()
    result["source"]["symbol"] = "BTC" * 200
    result["source"]["timeframe"] = "H1" * 200

    json_path, markdown_path = write_reports(result, tmp_path)

    assert len(json_path.name) <= reporting.MAX_REPORT_FILENAME_LENGTH
    assert len(markdown_path.name) <= reporting.MAX_REPORT_FILENAME_LENGTH
    assert "_abcdef12_20260719_123456" in json_path.name
    assert json_path.stem == markdown_path.stem


def test_write_reports_rejects_existing_final_directory_before_creating_temp_files(
    tmp_path: Path,
) -> None:
    result = _result()
    final_json_directory = tmp_path / "BTC_USDT_H1_test_abcdef12_20260719_123456.json"
    final_json_directory.mkdir()

    with pytest.raises(ValueError, match="JSON.*目錄"):
        write_reports(result, tmp_path)

    assert final_json_directory.is_dir()
    assert list(final_json_directory.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == [final_json_directory.name]


def test_write_reports_rejects_dangling_final_symlink_before_creating_temp_files(
    tmp_path: Path,
) -> None:
    result = _result()
    final_json_link = tmp_path / "BTC_USDT_H1_test_abcdef12_20260719_123456.json"
    try:
        final_json_link.symlink_to(tmp_path / "missing-report.json")
    except OSError as error:
        pytest.skip(f"目前 Windows 環境不允許建立符號連結：{error}")

    with pytest.raises(ValueError, match="JSON.*符號連結"):
        write_reports(result, tmp_path)

    assert final_json_link.is_symlink()
    assert sorted(path.name for path in tmp_path.iterdir()) == [final_json_link.name]


def test_write_reports_rolls_back_final_and_temp_files_when_markdown_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = reporting.os.replace

    def fail_markdown_publish(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.suffix == ".tmp" and destination_path.suffix == ".md":
            raise OSError("injected markdown publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_markdown_publish)

    with pytest.raises(OSError, match="injected markdown publish failure"):
        write_reports(_result(), tmp_path)

    assert list(tmp_path.iterdir()) == []
