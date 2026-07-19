from __future__ import annotations

from pathlib import Path

import pytest

import judge_strategy


def _result(status: str) -> dict:
    return {
        "verdict": {"status": status, "rules": []},
        "source": {
            "symbol": "BTCUSDT",
            "timeframe": "H1",
            "strategy_sha256": "abcdef123",
        },
        "created_at_utc": "2026-07-19T00:00:00+00:00",
        "full_sample": {},
        "cost_stress": {},
        "temporal_blocks": {},
        "regression": {},
        "limitations": {"true_out_of_sample": "僅為偽樣本外驗證"},
    }


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [("PASS", 0), ("REVIEW", 2), ("FAIL", 3)],
)
def test_main_writes_reports_and_returns_verdict_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: str, expected_code: int
) -> None:
    monkeypatch.setattr(judge_strategy, "run_judgment", lambda *args, **kwargs: _result(status))
    monkeypatch.setattr(
        judge_strategy,
        "write_reports",
        lambda result, output: (Path(output) / "report.json", Path(output) / "report.md"),
    )

    code = judge_strategy.main(
        [
            "--strategy",
            "strategy.json",
            "--data",
            "BTCUSDT_H1.parquet",
            "--output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert code == expected_code
    assert f"審判結果：{status}" in output
    assert "JSON 報告：" in output
    assert "Markdown 報告：" in output


@pytest.mark.parametrize(
    "argument, value",
    [
        ("--base-cost", "-0.0001"),
        ("--equal-blocks", "0"),
        ("--walk-forward-splits", "0"),
        ("--embargo-bars", "-1"),
    ],
)
def test_main_rejects_invalid_numeric_input_before_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    value: str,
) -> None:
    def must_not_run(*args: object, **kwargs: object) -> dict:
        raise AssertionError("不應執行審判核心")

    monkeypatch.setattr(judge_strategy, "run_judgment", must_not_run)

    code = judge_strategy.main(
        ["--strategy", "strategy.json", "--data", "BTCUSDT_H1.parquet", argument, value]
    )

    assert code == 1
    assert "輸入錯誤" in capsys.readouterr().err


def test_main_returns_one_for_execution_or_report_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        judge_strategy,
        "run_judgment",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("策略檔無效")),
    )

    code = judge_strategy.main(["--strategy", "bad.json", "--data", "missing.parquet"])

    assert code == 1
    assert "執行失敗：策略檔無效" in capsys.readouterr().err


def test_main_returns_one_for_unknown_verdict_without_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(judge_strategy, "run_judgment", lambda *args, **kwargs: _result("UNKNOWN"))
    monkeypatch.setattr(
        judge_strategy,
        "write_reports",
        lambda result, output: (Path(output) / "report.json", Path(output) / "report.md"),
    )

    code = judge_strategy.main(
        ["--strategy", "strategy.json", "--data", "BTCUSDT_H1.parquet", "--output-dir", str(tmp_path)]
    )

    assert code == 1
    assert "未知的審判結果" in capsys.readouterr().err
