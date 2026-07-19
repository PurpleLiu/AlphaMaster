# AlphaMaster Telegram 保存與策略審判設計

日期：2026-07-19  
狀態：已核准，待實作

## 1. 背景與目標

本次工作包含三個彼此相關、但須分開提交的項目：

1. 建立 PA_Agent 與 AlphaMaster 的自訂 Git 基準，保存目前台灣正體中文、加密貨幣、Azure OpenAI 與 Telegram 等既有修改。
2. 修正 AlphaMaster Telegram 設定在關閉或重新啟動後消失的問題。
3. 新增獨立的策略審判工具，驗證 AlphaMaster 找出的因子是否在成本、時間切分與基準市場風險調整後仍有可用價值。

AlphaMaster 現有 BTCUSDT H1 訓練繼續執行，不因本次工作中止。審判工具先驗證當前最佳策略快照，訓練結束後再對最終策略重跑。

## 2. Git 與資料安全

兩個專案採相同版本治理：

- `origin` 指向使用者 Fork：`PurpleLiu/PA_Agent` 與 `PurpleLiu/AlphaMaster`。
- 原作者倉庫保留為 `upstream`，供後續比較與同步。
- 自訂工作固定在 `custom/taiwan-crypto` 分支。
- 現有成果先做一個基準快照；Telegram 修正、審判工具與後續功能各自使用獨立提交。
- 本機設定、API Token、K 線資料、訓練 checkpoint、產生的策略與執行記錄不得提交。

Git 只保存可重建的程式碼與文件，不保存交易帳號憑證或執行期產物。

## 3. Telegram 設定保存

### 3.1 使用者體驗

AlphaMaster 即時分析頁面的 Telegram 區塊保留：

- 啟用開關
- Bot Token
- Chat ID
- 發送測試訊息
- 明確的「儲存設定」按鈕與保存狀態提示

欄位變更後以 500 毫秒 debounce 自動保存；按下儲存按鈕則立即保存。頁面重新載入及服務重新啟動時，後端讀取本機設定並回填所有欄位。

### 3.2 儲存位置與安全界線

- 設定寫入專案根目錄的 `web_settings.json`。
- 此檔已由 `.gitignore` 排除，Bot Token 不會推送至 GitHub。
- Token 目前採本機明文保存，以符合單機使用情境與既有設定架構。
- API 回傳 Token 是為了讓介面重載後能回填；介面必須使用密碼欄位顯示。
- 空 Token 或空 Chat ID 不得被視為可啟用的 Telegram 通知設定。

### 3.3 資料流

1. 頁面載入時呼叫 `GET /api/realtime/telegram`。
2. 後端以 `load_settings()` 讀取 `web_settings.json`，回傳 enable、token、chat ID。
3. 使用者修改欄位後，前端 debounce 呼叫 `PUT /api/realtime/telegram`。
4. 使用者按下儲存時，前端立即呼叫同一端點。
5. 後端只更新 Telegram 相關鍵值，避免覆寫 AI、資料檔案或其他設定。
6. 儲存成功後，前端顯示成功狀態；失敗時顯示錯誤且保留輸入內容。

### 3.4 驗證

測試至少涵蓋：

- Telegram 設定寫入後，以新的讀取流程可取得相同內容。
- 更新 Telegram 欄位時不會清除其他設定。
- 空 Token 或空 Chat ID 時，通知功能不會被判定為有效。
- API GET/PUT 的輸入輸出契約正確。
- 前端存在明確儲存操作、debounce 自動保存及頁面載入回填。

## 4. 策略審判工具

### 4.1 定位

審判工具不接受 AlphaMaster 訓練分數作為策略有效性的證據。它獨立讀取：

- Parquet K 線資料
- 策略 JSON
- AlphaMaster 的特徵計算與 StackVM 公式執行能力

工具重建訊號、部位與報酬，輸出可機器讀取的 JSON，以及台灣正體中文 Markdown 報告。

建議模組：

- `judgment/core.py`：純計算、指標、成本、切分與判定規則
- `judgment/runner.py`：載入資料與策略、執行 StackVM、組合報告資料
- `judge_strategy.py`：命令列入口
- `tests/unit/test_judgment.py`：核心規則單元測試

### 4.2 可追溯性

每次執行必須記錄：

- 策略來源路徑
- 策略檔 SHA-256
- 策略 JSON 快照或完整 token/formula 表示
- 資料來源、品種、週期、起訖時間與 K 線數量
- 執行參數與成本假設
- 程式執行時間與版本資訊

這可避免訓練持續更新 `best_BTCUSDT.json` 時，報告無法對應到當時被驗證的策略。

### 4.3 成本與報酬

基準單邊交易成本為 0.06%，並執行 1x、2x、3x、5x 成本壓力測試。成本在部位變動時扣除，以 turnover 為依據，不只在完整來回交易時計算。

核心績效包含：

- 累積與年化報酬
- Sharpe 與 Sortino
- 最大回撤
- Profit Factor
- turnover 與交易／換倉頻率
- 曝險比例、平均絕對部位、做多／做空拆分

### 4.4 穩定性與集中度

依自然年度及等長時間區塊計算績效，重點使用中位數而非平均數，並統計正報酬區塊比例。

集中度至少檢查：

- 最佳年度或最佳區塊對總收益的貢獻
- 前幾個最佳區塊的貢獻比例
- 移除最佳區塊後策略是否仍為正

若大部分收益只來自極少數時段，報告須標示嚴重集中風險。

### 4.5 BTC 基準風險調整

以 BTC 買入持有報酬為市場基準，對策略報酬進行 OLS：

- beta
- 年化 alpha
- 殘差 Sharpe
- 與 BTC 報酬的相關性

若策略主要只是帶方向的 BTC 曝險，而沒有正 alpha，不應被視為可獨立使用的套利或量化優勢。

### 4.6 時間切分與限制

執行多段前推式 pseudo walk-forward 驗證，切分之間加入 embargo，避免相鄰樣本因滾動特徵而直接污染。

由於同一歷史資料曾參與策略搜尋，這些結果只能稱為偽樣本外驗證，不能宣稱是真正的 out-of-sample。真正樣本外證據必須來自：

- 策略凍結後的新市場資料，或
- 連續 dry-run／paper trading 記錄

此限制必須出現在 Markdown 與 JSON 報告。

### 4.7 判定規則

最終狀態為 `PASS`、`REVIEW` 或 `FAIL`。`PASS` 至少需要同時滿足：

- Profit Factor >= 1.30
- Sharpe >= 0.75
- 2x 成本壓力下累積淨報酬不為負
- 時間區塊中位數報酬為正
- 至少 60% 時間區塊為正
- 年化 alpha 為正
- 沒有嚴重收益集中

`FAIL` 用於明確不符核心經濟性或穩定性條件；介於兩者、樣本不足或有需人工解讀的風險時標為 `REVIEW`。報告須逐條列出通過、失敗與原因，不能只給單一總分。

## 5. 錯誤處理

以下情況以清楚的台灣正體中文訊息終止，且 CLI 回傳非零狀態碼：

- 找不到或無法解析策略檔
- 找不到資料檔或缺少必要欄位
- 策略 token 無法由目前 StackVM 執行
- 有效資料過少，無法計算指定切分或風險指標
- 輸出路徑無法建立或寫入

單一指標因分母為零等原因無法計算時，不得靜默轉為零；應輸出 `null` 並附原因，判定通常降為 `REVIEW`。

## 6. 輸出與使用流程

預設輸出到不含憑證的報告目錄，檔名包含品種、週期、策略雜湊短碼與時間戳：

- `reports/judgment/*.json`
- `reports/judgment/*.md`

目前流程：

1. AlphaMaster 訓練繼續執行。
2. 對當前最佳策略快照執行一次審判，建立基準。
3. 訓練結束後對最終最佳策略重新執行。
4. 只有 `PASS` 或經人工確認的 `REVIEW` 才能進入 Freqtrade dry-run；不得直接啟用真實下單。
5. 累積新的未參與搜尋資料後，再執行真正的樣本外評估。

## 7. 非本次範圍

- 不在本次直接串接幣安真實下單。
- 不以槓桿放大回測績效。
- 不把 PA_Agent 的 LLM 判斷混入 AlphaMaster 審判結果。
- 不改變正在執行的 AlphaMaster 訓練參數或停止訓練。
- 不將 Telegram Token 或其他憑證改存到 Git 追蹤檔案。
