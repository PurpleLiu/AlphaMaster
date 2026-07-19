from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from judgment.runner import load_strategy_snapshot, run_judgment
from model_core.vocab import FORMULA_VOCAB


@pytest.fixture
def btc_h1_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import config

    monkeypatch.setattr(config.Config, "MIN_BARS", 100)
    bars = 240
    time = np.arange(bars, dtype=np.int64) * 3600 + 1_600_000_000
    close = 10_000 * np.exp(np.linspace(0, 0.10, bars))
    frame = pd.DataFrame(
        {
            "time": time,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(bars, 100.0),
        }
    )
    path = tmp_path / "BTCUSDT_H1.parquet"
    frame.to_parquet(path)
    return path


def test_load_strategy_snapshot_records_exact_sha256(tmp_path: Path) -> None:
    path = tmp_path / "best_BTCUSDT.json"
    path.write_text(
        json.dumps({"symbol": "BTCUSDT", "formula": [0]}), encoding="utf-8"
    )

    result = load_strategy_snapshot(path)

    assert result["resolved_path"] == str(path.resolve())
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["payload"]["formula"] == [0]
    assert result["formula"] == [0]


def test_run_judgment_executes_valid_formula_on_parquet(
    tmp_path: Path, btc_h1_parquet: Path
) -> None:
    strategy_path = tmp_path / "best_BTCUSDT.json"
    strategy_path.write_text(
        json.dumps({"symbol": "BTCUSDT", "formula": [0]}), encoding="utf-8"
    )

    result = run_judgment(
        strategy_path,
        btc_h1_parquet,
        equal_blocks=4,
        walk_forward_splits=4,
        embargo_bars=2,
    )

    assert result["schema_version"] == "1.0"
    assert result["source"]["strategy_sha256"]
    assert result["source"]["formula"] == [0]
    assert "strategy_payload" not in result["source"]
    assert result["source"]["symbol"] == "BTCUSDT"
    assert result["source"]["timeframe"] == "H1"
    assert result["source"]["bars"] == 240
    assert result["assumptions"]["base_one_way_cost"] == 0.0006
    assert set(result["cost_stress"]) == {"1x", "2x", "3x", "5x"}
    assert result["full_sample"]["total_turnover"] is not None
    assert result["limitations"]["is_true_out_of_sample"] is False
    assert result["limitations"]["pseudo_walk_forward"]
    assert result["verdict"]["status"] in {"PASS", "REVIEW", "FAIL"}


def test_run_judgment_excludes_untrusted_strategy_payload_secrets(
    tmp_path: Path, btc_h1_parquet: Path
) -> None:
    secret_values = ["runner-api-secret", "nested-token-secret", "private-chat-id"]
    strategy_path = tmp_path / "best_BTCUSDT.json"
    strategy_path.write_text(
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "formula": [0],
                "best_score": 1.25,
                "api_key": secret_values[0],
                "metadata": {
                    "token": secret_values[1],
                    "chat_id": secret_values[2],
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_judgment(
        strategy_path,
        btc_h1_parquet,
        equal_blocks=4,
        walk_forward_splits=4,
        embargo_bars=2,
    )

    source = result["source"]
    assert source["formula"] == [0]
    assert source["strategy_metadata"] == {"symbol": "BTCUSDT", "best_score": 1.25}
    assert "strategy_payload" not in source
    serialized = json.dumps(result, ensure_ascii=False)
    for secret in secret_values:
        assert secret not in serialized


def test_run_judgment_preserves_unavailable_metric_reasons(
    tmp_path: Path, btc_h1_parquet: Path
) -> None:
    strategy_path = tmp_path / "best_BTCUSDT.json"
    strategy_path.write_text(json.dumps([0]), encoding="utf-8")

    result = run_judgment(strategy_path, btc_h1_parquet)

    assert "unavailable_reasons" in result["full_sample"]
    assert "reason" in result["regression"]
    assert result["limitations"]["pseudo_walk_forward"] == result["walk_forward"][
        "limitation"
    ]


def test_run_judgment_rejects_invalid_stackvm_formula(
    tmp_path: Path, btc_h1_parquet: Path
) -> None:
    strategy_path = tmp_path / "best_BTCUSDT.json"
    strategy_path.write_text(json.dumps({"formula": [65]}), encoding="utf-8")

    with pytest.raises(ValueError, match="StackVM"):
        run_judgment(strategy_path, btc_h1_parquet)


@pytest.mark.parametrize("payload", [{}, {"formula": []}, {"formula": ["0"]}])
def test_load_strategy_snapshot_requires_non_empty_integer_formula(
    tmp_path: Path, payload: dict[object, object]
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="formula"):
        load_strategy_snapshot(path)


@pytest.mark.parametrize("token", [-1, FORMULA_VOCAB.size])
def test_load_strategy_snapshot_rejects_token_outside_current_vocab(
    tmp_path: Path, token: int
) -> None:
    path = tmp_path / "invalid-token.json"
    path.write_text(json.dumps({"formula": [token]}), encoding="utf-8")

    with pytest.raises(ValueError, match="token"):
        load_strategy_snapshot(path)
