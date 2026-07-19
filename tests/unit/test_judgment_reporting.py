from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
