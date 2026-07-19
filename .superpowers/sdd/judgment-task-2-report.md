# AlphaMaster Strategy Judgment CLI — Task 2 報告

日期：2026-07-19

## 完成內容

- 在 `judgment/core.py` 實作自然年度與等長時間區塊診斷，包含區塊起訖 UTC 時間、bars、報酬、中位數與正報酬比例。
- 實作區塊報酬集中度檢查：最佳區塊占正報酬比例、前三區塊占比與移除最佳區塊後報酬，並提供臺灣繁體中文原因。
- 實作市場 OLS 回歸：先移除非有限策略／基準配對，至少需要 30 筆資料；輸出 alpha、年化 alpha、beta、殘差 Sharpe 與相關性。資料不足或基準無變異時，所有不適用指標皆為 `None` 並附中文原因。
- 實作 pseudo walk-forward：使用 `np.array_split`，第二折起移除指定 embargo bars，對保留資料呼叫既有 `performance_metrics()`，並固定聲明不是實際樣本外驗證。
- 沿用 Task 1 的有限值驗證、正值期數檢查及 `performance_metrics()`；未加入任何模型訓練、資料處理或 UI 功能。

## 提交

- `a26e185 feat: add temporal and beta-adjusted diagnostics`

## 驗證

```text
.venv\Scripts\python -X utf8 -m pytest tests\unit\test_judgment_core.py \
  -k "temporal_blocks or concentration or market_regression or pseudo_walk_forward" -v
4 passed, 21 deselected

.venv\Scripts\python -X utf8 -m pytest tests\unit\test_judgment_core.py \
  -k "not (temporal_blocks or concentration or market_regression or pseudo_walk_forward)" -v
21 passed, 4 deselected

.venv\Scripts\python -X utf8 -m compileall -q judgment tests\unit
exit code 0
```

## Review follow-up (2026-07-19)

- `market_regression()` returns all unavailable diagnostics as `None`, with Traditional Chinese reasons, for a constant finite strategy or benchmark and for zero residual standard deviation.
- Added coverage for natural-year blocks, NaN/inf concentration rejection, finite-pair filtering, fewer than 30 pairs, both constant inputs, residual Sharpe, pseudo-WF limitation text, and a fully embargoed fold.

```text
.venv\Scripts\python -X utf8 -m pytest tests\unit\test_judgment_core.py -k "market_regression or temporal_blocks or concentration or pseudo_walk_forward" -v
13 passed, 21 deselected

.venv\Scripts\python -X utf8 -m pytest tests\unit\test_judgment_core.py -v
34 passed

.venv\Scripts\python -X utf8 -m compileall -q judgment tests\unit
exit code 0
```

## 注意事項

- `requirements.txt` 已經宣告 `pandas>=2.0.0`；本 worktree 的 `.venv` 原先未安裝 pandas，因此為執行指定 UTC 轉換補裝 `pandas 3.0.3`。未修改依賴宣告檔。
- pseudo walk-forward 會明確輸出「歷史資料曾參與策略搜尋；此結果僅為偽樣本外驗證。」；它不構成真正 out-of-sample 證據。
