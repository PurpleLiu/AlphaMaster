# AlphaMaster Strategy Judgment CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立獨立於訓練分數的 AlphaMaster 策略審判 CLI，對策略執行完整成本、時間穩定性、收益集中度、BTC beta/alpha 與偽樣本外檢驗，輸出 JSON 與台灣正體中文 Markdown。

**Architecture:** `judgment/core.py` 只處理 NumPy 陣列與統計判定；`judgment/runner.py` 負責策略、Parquet、StackVM 與報告資料組裝；`judgment/reporting.py` 負責序列化與台灣中文報告；`judge_strategy.py` 僅解析 CLI 參數及決定退出碼。所有經濟性規則都有合成資料單元測試，執行層另以臨時 Parquet 和合法簡單公式做整合測試。

**Tech Stack:** Python 3.12、NumPy、pandas、PyArrow、PyTorch、pytest、AlphaMaster `ParquetDataManager`、`MT5FeatureEngineer`、`StackVM`。

## Global Constraints

- 不使用 `best_score` 或訓練 reward 判定策略是否有效；僅把它列為來源資訊。
- 基準單邊成本為 0.0006，壓力倍數固定提供 1x、2x、3x、5x。
- `PASS` 必須同時滿足 PF >= 1.30、Sharpe >= 0.75、2x 成本淨報酬 >= 0、區塊中位數 > 0、正區塊比例 >= 0.60、年化 alpha > 0、無嚴重集中。
- 相同歷史資料曾參與策略搜尋，因此報告必須標示為 pseudo out-of-sample，不得宣稱真正 OOS。
- 無法計算的數值輸出 JSON `null` 並附原因，不得靜默替換成 0。
- 報告與 CLI 訊息使用台灣正體中文。
- 產生的報告、策略快照、Parquet 與 checkpoint 不得提交 Git。
- 不停止或修改正在執行的 AlphaMaster 訓練。

## File Map

- Create: `judgment/__init__.py` — 套件公開介面。
- Create: `judgment/core.py` — 報酬、績效、區塊、集中度、OLS、前推切分與 PASS/REVIEW/FAIL。
- Create: `judgment/runner.py` — 載入策略與資料、SHA-256、StackVM、成本情境與完整結果。
- Create: `judgment/reporting.py` — JSON 安全轉換、Markdown 與檔案輸出。
- Create: `judge_strategy.py` — CLI。
- Modify: `.gitignore` — 忽略 `reports/judgment/`。
- Create: `tests/unit/test_judgment_core.py` — 純統計與判定。
- Create: `tests/unit/test_judgment_runner.py` — 檔案、StackVM 與來源追溯整合。
- Create: `tests/unit/test_judgment_reporting.py` — JSON/Markdown 契約。

---

### Task 1: Build cost-aware return and performance primitives

**Files:**
- Create: `judgment/__init__.py`
- Create: `judgment/core.py`
- Create: `tests/unit/test_judgment_core.py`

**Interfaces:**
- Produces: `build_net_returns(position, market_return, one_way_cost) -> tuple[np.ndarray, np.ndarray]`
- Produces: `performance_metrics(net_returns, position, periods_per_year, turnover=None) -> dict[str, Any]`
- Produces: `cost_stress(position, market_return, base_cost, multipliers, periods_per_year) -> dict[str, dict]`

- [ ] **Step 1: Write failing tests for cost accounting and core metrics**

```python
from __future__ import annotations

import numpy as np

from judgment.core import build_net_returns, cost_stress, performance_metrics


def test_build_net_returns_charges_every_position_change() -> None:
    position = np.array([0.0, 1.0, 1.0, -1.0, 0.0])
    market = np.array([0.0, 0.01, -0.005, 0.02, 0.0])
    net, turnover = build_net_returns(position, market, one_way_cost=0.001)
    np.testing.assert_allclose(turnover, [0.0, 1.0, 0.0, 2.0, 1.0])
    np.testing.assert_allclose(net, position * market - turnover * 0.001)


def test_performance_metrics_include_pf_drawdown_and_exposure() -> None:
    pnl = np.array([0.01, -0.005, 0.02, -0.002])
    pos = np.array([1.0, 1.0, -0.5, 0.0])
    metrics = performance_metrics(pnl, pos, periods_per_year=365)
    assert abs(metrics["profit_factor"] - (0.03 / 0.007)) < 1e-12
    assert metrics["total_log_return"] == 0.023
    assert metrics["max_drawdown"] < 0
    assert metrics["exposure_ratio"] == 0.75
    assert metrics["long_exposure_ratio"] == 0.5
    assert metrics["short_exposure_ratio"] == 0.25


def test_performance_metrics_report_turnover_when_supplied() -> None:
    pnl = np.array([0.0, 0.01, -0.002])
    pos = np.array([0.0, 1.0, 0.0])
    turnover = np.array([0.0, 1.0, 1.0])
    metrics = performance_metrics(pnl, pos, 365, turnover=turnover)
    assert metrics["total_turnover"] == 2.0
    assert metrics["turnover_per_bar"] == 2.0 / 3.0
    assert metrics["rebalance_count"] == 2


def test_cost_stress_uses_fixed_multiplier_labels() -> None:
    position = np.array([0.0, 1.0, 1.0, 0.0])
    market = np.array([0.0, 0.01, 0.01, 0.0])
    result = cost_stress(position, market, 0.001, (1, 2, 3, 5), 365)
    assert list(result) == ["1x", "2x", "3x", "5x"]
    assert result["1x"]["total_log_return"] > result["5x"]["total_log_return"]
```

- [ ] **Step 2: Run tests and verify imports fail**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'judgment'`.

- [ ] **Step 3: Implement the numerical primitives**

Create `judgment/__init__.py` with exports for `JudgmentThresholds`, `classify_judgment`, and `run_judgment` once those modules exist. Initially export the three functions implemented in this task.

Implement `judgment/core.py` with these exact public signatures:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def build_net_returns(
    position: np.ndarray,
    market_return: np.ndarray,
    one_way_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    ret = np.asarray(market_return, dtype=np.float64).reshape(-1)
    if pos.shape != ret.shape:
        raise ValueError("position 與 market_return 長度必須相同")
    previous = np.concatenate(([0.0], pos[:-1]))
    turnover = np.abs(pos - previous)
    return pos * ret - turnover * float(one_way_cost), turnover


def performance_metrics(
    net_returns: np.ndarray,
    position: np.ndarray,
    periods_per_year: int,
    turnover: np.ndarray | None = None,
) -> dict[str, object]:
    pnl = np.asarray(net_returns, dtype=np.float64).reshape(-1)
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    if len(pnl) == 0 or len(pnl) != len(pos):
        raise ValueError("績效序列不可為空且長度必須一致")
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=0))
    downside = pnl[pnl < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else None
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 1e-12 else None
    sortino = (
        mean / downside_std * math.sqrt(periods_per_year)
        if downside_std is not None and downside_std > 1e-12 else None
    )
    curve = np.cumsum(pnl)
    drawdown = curve - np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:]
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    profit_factor = gains / losses if losses > 1e-12 else None
    years = len(pnl) / float(periods_per_year)
    total_log = float(pnl.sum())
    turn = None if turnover is None else np.asarray(turnover, dtype=np.float64).reshape(-1)
    if turn is not None and len(turn) != len(pnl):
        raise ValueError("turnover 與績效序列長度必須一致")
    unavailable: dict[str, str] = {}
    if sharpe is None:
        unavailable["sharpe"] = "報酬標準差為零"
    if sortino is None:
        unavailable["sortino"] = "沒有足夠的負報酬或下行標準差為零"
    if profit_factor is None:
        unavailable["profit_factor"] = "沒有可計算的負報酬"
    return {
        "bars": int(len(pnl)),
        "total_log_return": total_log,
        "total_return": float(np.expm1(total_log)),
        "annual_log_return": total_log / years if years > 0 else None,
        "annual_return": float(np.expm1(total_log / years)) if years > 0 else None,
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
```

Implement `cost_stress()` by calling `build_net_returns()` and `performance_metrics(..., turnover=turnover)` for each multiplier and adding `one_way_cost` to each scenario.

- [ ] **Step 4: Run core tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit the primitives**

```powershell
git add judgment/__init__.py judgment/core.py tests/unit/test_judgment_core.py
git commit -m "feat: add cost-aware judgment metrics"
```

---

### Task 2: Add temporal blocks, concentration, OLS, and pseudo walk-forward

**Files:**
- Modify: `judgment/core.py`
- Modify: `tests/unit/test_judgment_core.py`

**Interfaces:**
- Produces: `temporal_blocks(net_returns, times, periods_per_year, equal_blocks=8) -> dict`
- Produces: `concentration_analysis(block_returns) -> dict`
- Produces: `market_regression(strategy_returns, benchmark_returns, periods_per_year) -> dict`
- Produces: `pseudo_walk_forward(net_returns, times, periods_per_year, splits=5, embargo_bars=24) -> dict`

- [ ] **Step 1: Add failing tests for stability and beta adjustment**

Append tests that use UTC Unix seconds:

```python
from judgment.core import (
    concentration_analysis,
    market_regression,
    pseudo_walk_forward,
    temporal_blocks,
)


def test_temporal_blocks_report_median_and_positive_ratio() -> None:
    times = np.arange(8, dtype=np.int64) * 3600 + 1_600_000_000
    pnl = np.array([0.01, 0.01, -0.01, -0.01, 0.02, 0.02, 0.01, 0.01])
    result = temporal_blocks(pnl, times, 8760, equal_blocks=4)
    assert result["equal"]["returns"] == [0.02, -0.02, 0.04, 0.02]
    assert result["equal"]["median_return"] == 0.02
    assert result["equal"]["positive_ratio"] == 0.75


def test_concentration_flags_single_block_dependency() -> None:
    result = concentration_analysis([0.60, -0.05, -0.05, -0.05])
    assert result["severe"] is True
    assert result["return_without_best_block"] < 0


def test_market_regression_recovers_beta_and_positive_alpha() -> None:
    benchmark = np.linspace(-0.01, 0.01, 200)
    strategy = 0.5 * benchmark + 0.0002
    result = market_regression(strategy, benchmark, periods_per_year=365)
    assert abs(result["beta"] - 0.5) < 1e-9
    assert result["annual_alpha"] > 0
    assert result["correlation"] > 0.99


def test_pseudo_walk_forward_applies_embargo_and_marks_limitation() -> None:
    pnl = np.full(100, 0.001)
    times = np.arange(100, dtype=np.int64) * 3600 + 1_600_000_000
    result = pseudo_walk_forward(pnl, times, 8760, splits=5, embargo_bars=2)
    assert len(result["folds"]) == 5
    assert result["folds"][1]["start_index"] == 22
    assert result["is_true_out_of_sample"] is False
```

- [ ] **Step 2: Run and verify missing functions**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py -v`

Expected: FAIL on imports for the four new functions.

- [ ] **Step 3: Implement temporal diagnostics**

Implementation requirements:

```python
def concentration_analysis(block_returns: Iterable[float]) -> dict[str, float | bool | None]:
    values = np.asarray(list(block_returns), dtype=np.float64)
    if len(values) < 2:
        return {"severe": True, "reason": "時間區塊不足", "best_block_share": None,
                "top_three_share": None, "return_without_best_block": None}
    total_positive = float(values[values > 0].sum())
    ordered = np.sort(values)[::-1]
    best_share = float(ordered[0] / total_positive) if total_positive > 1e-12 else None
    top_three = float(ordered[:3].clip(min=0).sum() / total_positive) if total_positive > 1e-12 else None
    without_best = float(values.sum() - ordered[0])
    severe = best_share is None or best_share > 0.50 or without_best <= 0
    return {"severe": severe, "reason": "收益過度集中" if severe else "未見嚴重集中",
            "best_block_share": best_share, "top_three_share": top_three,
            "return_without_best_block": without_best}
```

`temporal_blocks()` must:

- convert `times` with `pandas.to_datetime(times, unit="s", utc=True)`;
- aggregate natural-year sums;
- split `np.arange(len(net_returns))` with `np.array_split(..., equal_blocks)`;
- return each block's start/end timestamp, bars, return, median return, and positive ratio;
- pass equal-block returns to `concentration_analysis()`.

`market_regression()` must drop non-finite pairs, require at least 30 observations, use `np.linalg.lstsq(np.column_stack([ones, benchmark]), strategy, rcond=None)`, annualize the intercept by multiplication with `periods_per_year`, and compute residual Sharpe plus correlation. Insufficient or constant benchmark data returns `None` metrics with a Chinese reason.

`pseudo_walk_forward()` must use `np.array_split(np.arange(n), splits)`, remove the first `embargo_bars` observations from folds after the first, run `performance_metrics()` on each retained fold, and always return `is_true_out_of_sample=False` plus the limitation text `歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。`.

- [ ] **Step 4: Run all core tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py -v`

Expected: 8 passed.

- [ ] **Step 5: Commit temporal and market diagnostics**

```powershell
git add judgment/core.py tests/unit/test_judgment_core.py
git commit -m "feat: add temporal and beta-adjusted diagnostics"
```

---

### Task 3: Implement transparent PASS, REVIEW, and FAIL classification

**Files:**
- Modify: `judgment/core.py`
- Modify: `tests/unit/test_judgment_core.py`

**Interfaces:**
- Produces: `JudgmentThresholds` dataclass.
- Produces: `classify_judgment(full_metrics, stress, blocks, regression, concentration, thresholds=None) -> dict`.

- [ ] **Step 1: Add failing classification tests**

```python
from judgment.core import classify_judgment


def _passing_inputs():
    return (
        {"profit_factor": 1.5, "sharpe": 1.0},
        {"2x": {"total_log_return": 0.1}},
        {"equal": {"median_return": 0.02, "positive_ratio": 0.75}},
        {"annual_alpha": 0.1},
        {"severe": False},
    )


def test_classification_pass_requires_every_gate() -> None:
    result = classify_judgment(*_passing_inputs())
    assert result["status"] == "PASS"
    assert all(rule["passed"] is True for rule in result["rules"])


def test_classification_fail_for_negative_economic_edge() -> None:
    args = list(_passing_inputs())
    args[0] = {"profit_factor": 0.8, "sharpe": -0.2}
    result = classify_judgment(*args)
    assert result["status"] == "FAIL"


def test_classification_review_when_metric_is_unavailable() -> None:
    args = list(_passing_inputs())
    args[3] = {"annual_alpha": None, "reason": "樣本不足"}
    result = classify_judgment(*args)
    assert result["status"] == "REVIEW"
```

- [ ] **Step 2: Run and verify classification is absent**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py -k classification -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement explicit gates**

```python
@dataclass(frozen=True)
class JudgmentThresholds:
    min_profit_factor: float = 1.30
    min_sharpe: float = 0.75
    min_stress_2x_return: float = 0.0
    min_block_median: float = 0.0
    min_positive_block_ratio: float = 0.60
    min_annual_alpha: float = 0.0


def classify_judgment(full_metrics, stress, blocks, regression, concentration,
                      thresholds: JudgmentThresholds | None = None) -> dict:
    t = thresholds or JudgmentThresholds()
    checks = [
        ("profit_factor", full_metrics.get("profit_factor"), t.min_profit_factor, ">="),
        ("sharpe", full_metrics.get("sharpe"), t.min_sharpe, ">="),
        ("cost_stress_2x", stress.get("2x", {}).get("total_log_return"), t.min_stress_2x_return, ">="),
        ("block_median", blocks.get("equal", {}).get("median_return"), t.min_block_median, ">"),
        ("positive_block_ratio", blocks.get("equal", {}).get("positive_ratio"), t.min_positive_block_ratio, ">="),
        ("annual_alpha", regression.get("annual_alpha"), t.min_annual_alpha, ">"),
        ("concentration", not concentration.get("severe") if concentration.get("severe") is not None else None, True, "=="),
    ]
```

Convert checks to `rules` dictionaries containing `name`, `value`, `threshold`, `operator`, `passed`, and a Traditional Chinese explanation. If any value is `None`, status is `REVIEW`. Otherwise status is `PASS` when all pass; if PF < 1.0, Sharpe <= 0, or 2x return < 0, status is `FAIL`; remaining failed gates produce `REVIEW` for manual assessment.

- [ ] **Step 4: Run classification and full core tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py -v`

Expected: 11 passed.

- [ ] **Step 5: Commit classification rules**

```powershell
git add judgment/core.py tests/unit/test_judgment_core.py
git commit -m "feat: classify strategy judgment outcomes"
```

---

### Task 4: Load a strategy snapshot and reproduce AlphaMaster positions

**Files:**
- Create: `judgment/runner.py`
- Create: `tests/unit/test_judgment_runner.py`

**Interfaces:**
- Produces: `load_strategy_snapshot(path: str | Path) -> dict`.
- Produces: `run_judgment(strategy_path, data_path, base_cost=0.0006, stress_multipliers=(1,2,3,5), equal_blocks=8, walk_forward_splits=5, embargo_bars=24) -> dict`.
- Consumes: `ParquetDataManager`, `estimate_periods_per_year`, `StackVM`, `compute_target_positions_stateless`, and Task 1-3 core functions.

- [ ] **Step 1: Write failing runner tests**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from judgment.runner import load_strategy_snapshot, run_judgment


@pytest.fixture
def btc_h1_parquet(tmp_path: Path, monkeypatch) -> Path:
    import config
    monkeypatch.setattr(config.Config, "MIN_BARS", 100)
    bars = 240
    time = np.arange(bars, dtype=np.int64) * 3600 + 1_600_000_000
    close = 10_000 * np.exp(np.linspace(0, 0.10, bars))
    frame = pd.DataFrame({
        "time": time,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": np.full(bars, 100.0),
    })
    path = tmp_path / "BTCUSDT_H1.parquet"
    frame.to_parquet(path)
    return path


def test_load_strategy_snapshot_records_exact_sha256(tmp_path: Path) -> None:
    path = tmp_path / "best_BTCUSDT.json"
    path.write_text(json.dumps({"symbol": "BTCUSDT", "formula": [0]}), encoding="utf-8")
    result = load_strategy_snapshot(path)
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["payload"]["formula"] == [0]


def test_run_judgment_executes_valid_formula_on_parquet(
    tmp_path: Path, btc_h1_parquet: Path
) -> None:
    strategy_path = tmp_path / "best_BTCUSDT.json"
    strategy_path.write_text(json.dumps({"symbol": "BTCUSDT", "formula": [0]}), encoding="utf-8")

    result = run_judgment(strategy_path, btc_h1_parquet, equal_blocks=4,
                          walk_forward_splits=4, embargo_bars=2)
    assert result["source"]["strategy_sha256"]
    assert result["source"]["bars"] == 240
    assert result["assumptions"]["base_one_way_cost"] == 0.0006
    assert set(result["cost_stress"]) == {"1x", "2x", "3x", "5x"}
    assert result["limitations"]["is_true_out_of_sample"] is False


def test_run_judgment_rejects_invalid_stackvm_formula(
    tmp_path: Path, btc_h1_parquet: Path
) -> None:
    strategy_path = tmp_path / "best_BTCUSDT.json"
    strategy_path.write_text(json.dumps({"symbol": "BTCUSDT", "formula": [65]}), encoding="utf-8")
    with pytest.raises(ValueError, match="StackVM 無法執行"):
        run_judgment(strategy_path, btc_h1_parquet)
```

- [ ] **Step 2: Run tests and verify runner is absent**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_runner.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement source loading and position reconstruction**

`load_strategy_snapshot()` must read raw bytes once, compute SHA-256, decode UTF-8 JSON, accept either a token list or object, require a non-empty integer formula, and return resolved path, hash, and the exact payload.

`run_judgment()` must execute this sequence:

```python
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
```

Then compute baseline PnL, stress scenarios, temporal blocks, concentration, regression against `market_return`, pseudo walk-forward, and classification. Pass baseline turnover into `performance_metrics()` so full-sample output includes total turnover, turnover per bar, and rebalance count. The result must include `schema_version="1.0"`, UTC creation time, source paths/hash/payload/symbol/timeframe/bars/date range, all assumptions, diagnostics, verdict rules, unavailable-metric reasons, and the mandatory pseudo-OOS limitation.

- [ ] **Step 4: Run runner and core tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_runner.py tests/unit/test_judgment_core.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the AlphaMaster integration**

```powershell
git add judgment/runner.py tests/unit/test_judgment_runner.py judgment/__init__.py
git commit -m "feat: run independent judgment on AlphaMaster strategies"
```

---

### Task 5: Generate JSON and Taiwan Traditional Chinese Markdown reports

**Files:**
- Create: `judgment/reporting.py`
- Create: `tests/unit/test_judgment_reporting.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `to_json_safe(value) -> JSON-safe value`.
- Produces: `render_markdown(result: dict) -> str`.
- Produces: `write_reports(result, output_dir: str | Path) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing report tests**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from judgment.reporting import render_markdown, to_json_safe, write_reports


def test_to_json_safe_converts_numpy_and_nonfinite_to_null() -> None:
    result = to_json_safe({"x": np.float64(1.2), "bad": np.nan, "n": np.int64(3)})
    assert result == {"x": 1.2, "bad": None, "n": 3}


def test_markdown_contains_verdict_rules_and_oos_warning() -> None:
    result = {
        "verdict": {"status": "REVIEW", "rules": [
            {"name": "sharpe", "passed": False, "explanation": "Sharpe 未達門檻"}
        ]},
        "source": {"symbol": "BTCUSDT", "timeframe": "H1", "strategy_sha256": "abcdef"},
        "full_sample": {"sharpe": 0.5, "profit_factor": 1.1},
        "limitations": {"message": "歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。"},
        "cost_stress": {}, "temporal": {}, "market_regression": {},
    }
    text = render_markdown(result)
    assert "# AlphaMaster 策略審判報告" in text
    assert "REVIEW" in text
    assert "BTCUSDT" in text
    assert "偽樣本外" in text


def test_write_reports_uses_symbol_timeframe_hash_and_writes_valid_json(tmp_path: Path) -> None:
    result = {
        "created_at": "2026-07-19T12:34:56+00:00",
        "source": {"symbol": "BTCUSDT", "timeframe": "H1", "strategy_sha256": "abcdef123456"},
        "verdict": {"status": "REVIEW", "rules": []},
        "full_sample": {}, "cost_stress": {}, "temporal": {},
        "market_regression": {}, "limitations": {"message": "偽樣本外"},
    }
    json_path, md_path = write_reports(result, tmp_path)
    assert "BTCUSDT_H1_abcdef12" in json_path.name
    assert json.loads(json_path.read_text(encoding="utf-8"))["verdict"]["status"] == "REVIEW"
    assert md_path.read_text(encoding="utf-8").startswith("# AlphaMaster 策略審判報告")
```

- [ ] **Step 2: Run and verify report module is absent**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_reporting.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement deterministic report output**

`to_json_safe()` must recursively convert dictionaries, lists, NumPy scalars and arrays; non-finite floats become `None`.

`render_markdown()` must include these sections in order:

1. 標題與 PASS/REVIEW/FAIL。
2. 策略 SHA-256、策略路徑、資料路徑、品種、週期、日期與 bars。
3. 逐條門檻表格：規則、數值、門檻、結果、說明。
4. 全樣本績效。
5. 成本 1x/2x/3x/5x 壓力表。
6. 年度、等長區塊與集中度。
7. BTC beta、年化 alpha、殘差 Sharpe、相關性。
8. pseudo walk-forward 各 fold。
9. 粗體限制聲明：歷史資料曾參與策略搜尋，不是真正樣本外。

`write_reports()` must create the output directory, derive a filename from sanitized symbol/timeframe, first eight SHA characters and UTC `YYYYMMDD_HHMMSS`, write JSON with `ensure_ascii=False, indent=2`, and write Markdown as UTF-8.

Append this exact ignore rule to `.gitignore`:

```gitignore
# Local strategy judgment outputs
reports/judgment/
```

- [ ] **Step 4: Run report tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_reporting.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit report generation**

```powershell
git add judgment/reporting.py tests/unit/test_judgment_reporting.py .gitignore
git commit -m "feat: generate strategy judgment reports"
```

---

### Task 6: Add CLI behavior and machine-readable exit codes

**Files:**
- Create: `judge_strategy.py`
- Create: `tests/unit/test_judge_strategy_cli.py`

**Interfaces:**
- Consumes: `run_judgment()` and `write_reports()`.
- Produces: CLI exit 0 for PASS, 2 for REVIEW, 3 for FAIL, 1 for execution/input errors.

- [ ] **Step 1: Write failing CLI tests**

```python
from __future__ import annotations

from pathlib import Path

import judge_strategy


def test_main_writes_reports_and_returns_verdict_code(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(judge_strategy, "run_judgment", lambda *a, **k: {
        "verdict": {"status": "REVIEW", "rules": []},
        "source": {"symbol": "BTCUSDT", "timeframe": "H1", "strategy_sha256": "abcdef123"},
        "created_at": "2026-07-19T00:00:00+00:00",
        "full_sample": {}, "cost_stress": {}, "temporal": {},
        "market_regression": {}, "limitations": {"message": "偽樣本外"},
    })
    monkeypatch.setattr(judge_strategy, "write_reports", lambda result, output: (
        Path(output) / "report.json", Path(output) / "report.md"
    ))
    code = judge_strategy.main([
        "--strategy", "strategy.json", "--data", "BTCUSDT_H1.parquet",
        "--output-dir", str(tmp_path),
    ])
    assert code == 2


def test_main_returns_one_for_invalid_input(monkeypatch, capsys) -> None:
    monkeypatch.setattr(judge_strategy, "run_judgment", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("策略公式無效")
    ))
    code = judge_strategy.main(["--strategy", "bad.json", "--data", "missing.parquet"])
    assert code == 1
    assert "策略公式無效" in capsys.readouterr().err
```

- [ ] **Step 2: Run and verify CLI module is absent**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judge_strategy_cli.py -v`

Expected: FAIL on import.

- [ ] **Step 3: Implement argparse and explicit exit mapping**

`judge_strategy.py` must provide:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="獨立審判 AlphaMaster 策略")
    parser.add_argument("--strategy", required=True, help="策略 JSON 路徑")
    parser.add_argument("--data", required=True, help="本地 Parquet K 線路徑")
    parser.add_argument("--base-cost", type=float, default=0.0006, help="單邊成本，小數格式")
    parser.add_argument("--equal-blocks", type=int, default=8)
    parser.add_argument("--walk-forward-splits", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=24)
    parser.add_argument("--output-dir", default="reports/judgment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_judgment(
            args.strategy, args.data, base_cost=args.base_cost,
            equal_blocks=args.equal_blocks,
            walk_forward_splits=args.walk_forward_splits,
            embargo_bars=args.embargo_bars,
        )
        json_path, markdown_path = write_reports(result, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"審判失敗：{exc}", file=sys.stderr)
        return 1
    status = result["verdict"]["status"]
    print(f"審判結果：{status}")
    print(f"JSON：{json_path}")
    print(f"報告：{markdown_path}")
    return {"PASS": 0, "REVIEW": 2, "FAIL": 3}[status]
```

End the file with `raise SystemExit(main())` under the standard `if __name__ == "__main__"` guard.

- [ ] **Step 4: Run CLI tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judge_strategy_cli.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit the CLI**

```powershell
git add judge_strategy.py tests/unit/test_judge_strategy_cli.py
git commit -m "feat: add strategy judgment CLI"
```

---

### Task 7: Verify with the current BTCUSDT strategy snapshot

**Files:**
- Generated locally and ignored: `reports/judgment/*.json`
- Generated locally and ignored: `reports/judgment/*.md`

**Interfaces:**
- Verifies the complete CLI and records a reproducible local result.

- [ ] **Step 1: Run all judgment tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_judgment_core.py tests/unit/test_judgment_runner.py tests/unit/test_judgment_reporting.py tests/unit/test_judge_strategy_cli.py -v`

Expected: all judgment tests pass.

- [ ] **Step 2: Run the complete unit suite and syntax checks**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit -q`

Expected: all unit tests pass.

Run: `.venv\Scripts\python -X utf8 -m compileall -q judgment judge_strategy.py tests/unit`

Expected: exit code 0.

- [ ] **Step 3: Snapshot and judge the current best strategy**

Run:

```powershell
.venv\Scripts\python -X utf8 judge_strategy.py `
  --strategy strategies\best_BTCUSDT.json `
  --data "data\K線資料\BTCUSDT_H1.parquet" `
  --base-cost 0.0006 `
  --output-dir reports\judgment
```

Expected: CLI prints PASS, REVIEW, or FAIL plus JSON/Markdown paths. Exit 2 or 3 is a valid analytical outcome, not a software test failure.

- [ ] **Step 4: Inspect reproducibility and limitations**

Confirm in both output files:

- strategy SHA-256 matches `Get-FileHash strategies\best_BTCUSDT.json -Algorithm SHA256`;
- all 1x/2x/3x/5x costs are present;
- every PASS gate has an explicit result;
- BTC beta/alpha and concentration are present or carry a reason for `null`;
- the pseudo-OOS limitation is prominent;
- no API key, Telegram Token, Azure credential, or exchange credential appears.

- [ ] **Step 5: Verify generated outputs are ignored and push code**

Run: `git status --short --ignored reports/judgment`

Expected: report files are prefixed `!!` and are not staged.

Run: `git push origin custom/taiwan-crypto`

Expected: all code/test commits are pushed, while local reports remain untracked and ignored.
