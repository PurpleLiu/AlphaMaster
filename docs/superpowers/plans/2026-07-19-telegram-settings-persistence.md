# AlphaMaster Telegram Settings Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 AlphaMaster 的 Telegram 啟用狀態、Bot Token 與 Chat ID 在儲存、頁面重載及服務重啟後仍能正確還原，並提供按鈕儲存與 500ms debounce 自動儲存。

**Architecture:** `web.settings` 是唯一持久化邊界，負責正規化及原子更新 `web_settings.json`；FastAPI GET/PUT 只轉換 API 契約；前端共用同一個儲存函式，手動儲存立即送出，欄位變更則經 debounce 送出。通知模組使用同一個有效性函式，避免只有 enable、沒有憑證時被當成可用。

**Tech Stack:** Python 3.12、FastAPI、Pydantic 2、pytest、原生 JavaScript、JSON。

## Global Constraints

- `web_settings.json` 維持本機明文儲存並由 Git 忽略，不得提交真實 Bot Token。
- Telegram 欄位更新不得清除 Azure OpenAI、資料檔、回測成本或監控清單等其他設定。
- 空 Bot Token 或空 Chat ID 不得被視為有效通知設定。
- UI 與錯誤訊息使用台灣正體中文。
- 不停止或修改正在執行的 AlphaMaster 訓練。

## File Map

- Modify: `web/settings.py` — Telegram 欄位正規化、儲存及有效性判定。
- Modify: `web/app.py` — GET/PUT 回傳一致的 `active` 狀態。
- Modify: `web/telegram_notify.py` — 共用設定有效性判定。
- Modify: `web/static/index.html` — 台灣用語的明確儲存按鈕。
- Modify: `web/static/app.js` — 500ms debounce、自動儲存、手動儲存與狀態提示。
- Test: `tests/unit/test_telegram_settings.py` — 本機持久化與有效性。
- Test: `tests/unit/test_telegram_api.py` — FastAPI API 契約與跨重新讀取儲存。
- Test: `tests/unit/test_telegram_frontend_contract.py` — 前端必要元素與事件契約。

---

### Task 1: Persist and validate Telegram settings

**Files:**
- Modify: `web/settings.py:109-218`
- Modify: `web/telegram_notify.py:12-84`
- Create: `tests/unit/test_telegram_settings.py`

**Interfaces:**
- Produces: `telegram_is_active(settings: Mapping[str, Any]) -> bool`
- Consumes: existing `load_settings() -> dict` and `save_settings(data: dict) -> dict`

- [ ] **Step 1: Write failing persistence tests**

```python
from __future__ import annotations

import json

import web.settings as settings_mod
from web.settings import load_settings, save_settings, telegram_is_active


def test_save_and_reload_telegram_settings_without_losing_other_keys() -> None:
    settings_mod.SETTINGS_PATH.write_text(
        json.dumps({"ai_model": "model-router", "last_data_file": "BTCUSDT_H1.parquet"}),
        encoding="utf-8",
    )

    saved = save_settings({
        "telegram_enabled": True,
        "telegram_bot_token": " 123456789:AA-test-token ",
        "telegram_chat_id": " 688220938 ",
    })
    reloaded = load_settings()

    assert saved["telegram_enabled"] is True
    assert reloaded["telegram_bot_token"] == "123456789:AA-test-token"
    assert reloaded["telegram_chat_id"] == "688220938"
    assert reloaded["ai_model"] == "model-router"
    assert reloaded["last_data_file"] == "BTCUSDT_H1.parquet"
    persisted = json.loads(settings_mod.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert persisted["telegram_chat_id"] == "688220938"


def test_telegram_is_active_requires_enable_token_and_chat_id() -> None:
    base = {
        "telegram_enabled": True,
        "telegram_bot_token": "token",
        "telegram_chat_id": "chat",
    }
    assert telegram_is_active(base) is True
    assert telegram_is_active({**base, "telegram_enabled": False}) is False
    assert telegram_is_active({**base, "telegram_bot_token": ""}) is False
    assert telegram_is_active({**base, "telegram_chat_id": "  "}) is False


def test_load_settings_normalizes_telegram_values() -> None:
    settings_mod.SETTINGS_PATH.write_text(
        json.dumps({
            "telegram_enabled": 1,
            "telegram_bot_token": " token ",
            "telegram_chat_id": 12345,
        }),
        encoding="utf-8",
    )
    loaded = load_settings()
    assert loaded["telegram_enabled"] is True
    assert loaded["telegram_bot_token"] == "token"
    assert loaded["telegram_chat_id"] == "12345"
```

- [ ] **Step 2: Run tests and confirm the current bug**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_telegram_settings.py -v`

Expected: FAIL because `save_settings()` ignores the three `telegram_*` keys and `telegram_is_active` does not exist.

- [ ] **Step 3: Implement normalization, persistence, and active-state helper**

Add the import and helper to `web/settings.py`:

```python
from typing import Any, Mapping


def telegram_is_active(settings: Mapping[str, Any]) -> bool:
    return bool(settings.get("telegram_enabled")) and bool(
        str(settings.get("telegram_bot_token") or "").strip()
    ) and bool(str(settings.get("telegram_chat_id") or "").strip())
```

In `load_settings()`, immediately after Feishu normalization, add:

```python
    out["telegram_enabled"] = bool(out.get("telegram_enabled", False))
    out["telegram_bot_token"] = str(out.get("telegram_bot_token") or "").strip()
    out["telegram_chat_id"] = str(out.get("telegram_chat_id") or "").strip()
```

In `save_settings()`, before writing `SETTINGS_PATH`, add:

```python
    if "telegram_enabled" in data:
        current["telegram_enabled"] = bool(data["telegram_enabled"])
    if "telegram_bot_token" in data:
        current["telegram_bot_token"] = str(data["telegram_bot_token"] or "").strip()
    if "telegram_chat_id" in data:
        current["telegram_chat_id"] = str(data["telegram_chat_id"] or "").strip()
```

Change `web/telegram_notify.py` to import and use the helper:

```python
from web.settings import load_settings, telegram_is_active

# inside notify_direction_flip
    settings = load_settings()
    if not telegram_is_active(settings):
        return False, "Telegram 通知未啟用或憑證不完整"
```

- [ ] **Step 4: Run focused tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_telegram_settings.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit the persistence boundary**

```powershell
git add web/settings.py web/telegram_notify.py tests/unit/test_telegram_settings.py
git commit -m "fix: persist AlphaMaster Telegram settings"
```

---

### Task 2: Make the Telegram API contract expose persisted validity

**Files:**
- Modify: `web/app.py:1111-1158`
- Create: `tests/unit/test_telegram_api.py`

**Interfaces:**
- Consumes: `telegram_is_active(settings: Mapping[str, Any]) -> bool` from Task 1.
- Produces: GET and PUT JSON `{ok?, enabled, active, bot_token, chat_id}`.

- [ ] **Step 1: Write failing API tests**

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app


client = TestClient(app)


def test_put_then_get_telegram_settings_round_trip() -> None:
    put = client.put("/api/realtime/telegram", json={
        "enabled": True,
        "bot_token": " 123456789:AA-test-token ",
        "chat_id": " 688220938 ",
    })
    assert put.status_code == 200
    assert put.json() == {
        "ok": True,
        "enabled": True,
        "active": True,
        "bot_token": "123456789:AA-test-token",
        "chat_id": "688220938",
    }

    get = client.get("/api/realtime/telegram")
    assert get.status_code == 200
    assert get.json()["active"] is True
    assert get.json()["chat_id"] == "688220938"


def test_enabled_with_empty_credentials_is_not_active() -> None:
    response = client.put("/api/realtime/telegram", json={
        "enabled": True,
        "bot_token": "",
        "chat_id": "",
    })
    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["active"] is False
```

- [ ] **Step 2: Run tests and verify missing API field**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_telegram_api.py -v`

Expected: FAIL because GET/PUT do not return `active`.

- [ ] **Step 3: Add the shared validity field**

Import the helper in `web/app.py`:

```python
from web.settings import load_settings, save_settings, telegram_is_active
```

Use one response helper so GET and PUT cannot drift:

```python
def _telegram_settings_response(settings: dict[str, Any], *, ok: bool = False) -> dict[str, Any]:
    response = {
        "enabled": bool(settings.get("telegram_enabled")),
        "active": telegram_is_active(settings),
        "bot_token": settings.get("telegram_bot_token") or "",
        "chat_id": settings.get("telegram_chat_id") or "",
    }
    if ok:
        response["ok"] = True
    return response
```

Make GET return `_telegram_settings_response(load_settings())` and PUT return `_telegram_settings_response(saved, ok=True)` after its existing partial-update payload logic.

同時把 Telegram 測試端點的三段文字固定為可讀的台灣正體中文：缺 Token 回傳 `請先填寫 Bot Token`、缺 Chat ID 回傳 `請先填寫 Chat ID`，測試訊息為 `✅ AlphaMaster Telegram 測試成功，後續信號轉折會推送到這裡。`。

- [ ] **Step 4: Run API and settings tests**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_telegram_api.py tests/unit/test_telegram_settings.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit the API contract**

```powershell
git add web/app.py tests/unit/test_telegram_api.py
git commit -m "test: cover Telegram settings API round trip"
```

---

### Task 3: Add 500ms auto-save without weakening manual save

**Files:**
- Modify: `web/static/index.html:558-562`
- Modify: `web/static/app.js:1983-2047,2663-2665`
- Create: `tests/unit/test_telegram_frontend_contract.py`

**Interfaces:**
- Consumes: PUT `/api/realtime/telegram` from Task 2.
- Produces: `scheduleRtTelegramSave()` and `saveRtTelegramSettings({ automatic = false } = {})`.

- [ ] **Step 1: Write a failing static contract test**

```python
from __future__ import annotations

from pathlib import Path


APP_JS = Path(__file__).resolve().parents[2] / "web" / "static" / "app.js"
INDEX_HTML = Path(__file__).resolve().parents[2] / "web" / "static" / "index.html"


def test_telegram_frontend_has_debounced_and_manual_save() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "const RT_TELEGRAM_SAVE_DELAY_MS = 500;" in source
    assert "function scheduleRtTelegramSave()" in source
    assert "saveRtTelegramSettings({ automatic: true })" in source
    assert '["rtTelegramEnabled", "rtTelegramToken", "rtTelegramChatId"]' in source
    assert 'addEventListener("click", () => saveRtTelegramSettings())' in source
    assert 'id="rtTelegramSaveBtn">儲存設定</button>' in html
```

- [ ] **Step 2: Run the contract test and confirm it fails**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_telegram_frontend_contract.py -v`

Expected: FAIL because no debounce scheduler exists.

- [ ] **Step 3: Implement the shared save path and scheduler**

Add module-level state near the existing real-time UI functions:

```javascript
const RT_TELEGRAM_SAVE_DELAY_MS = 500;
let rtTelegramSaveTimer = null;

function scheduleRtTelegramSave() {
  if (rtTelegramSaveTimer) window.clearTimeout(rtTelegramSaveTimer);
  const hint = $("rtTelegramHint");
  if (hint) {
    hint.textContent = "設定已變更，準備自動儲存…";
    hint.classList.remove("valid", "bad", "invalid");
  }
  rtTelegramSaveTimer = window.setTimeout(() => {
    rtTelegramSaveTimer = null;
    saveRtTelegramSettings({ automatic: true });
  }, RT_TELEGRAM_SAVE_DELAY_MS);
}
```

Change the save signature and success copy:

```javascript
async function saveRtTelegramSettings({ automatic = false } = {}) {
  if (!automatic && rtTelegramSaveTimer) {
    window.clearTimeout(rtTelegramSaveTimer);
    rtTelegramSaveTimer = null;
  }
  // retain the existing PUT body and error handling
  // on success:
  hint.textContent = automatic ? "Telegram 設定已自動儲存。" : "Telegram 設定已儲存。";
}
```

Replace the Telegram event bindings with:

```javascript
  if ($("rtTelegramSaveBtn")) {
    $("rtTelegramSaveBtn").addEventListener("click", () => saveRtTelegramSettings());
  }
  if ($("rtTelegramTestBtn")) $("rtTelegramTestBtn").addEventListener("click", testRtTelegram);
  ["rtTelegramEnabled", "rtTelegramToken", "rtTelegramChatId"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener(id === "rtTelegramEnabled" ? "change" : "input", scheduleRtTelegramSave);
  });
```

In `web/static/index.html`, change the button label to:

```html
<button type="button" class="btn btn-primary" id="rtTelegramSaveBtn">儲存設定</button>
```

Replace every Telegram-specific mojibake status string in `loadRtTelegramSettings()`, `saveRtTelegramSettings()`, and `testRtTelegram()` with these exact messages:

- `載入 Telegram 設定失敗：${e.message}`
- `設定已變更，準備自動儲存…`
- `Telegram 設定已自動儲存。`
- `Telegram 設定已儲存。`
- `儲存失敗：${e.message}`
- `Telegram 測試訊息已送出。`
- `測試失敗：${e.message}`

- [ ] **Step 4: Run the focused suite**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit/test_telegram_frontend_contract.py tests/unit/test_telegram_api.py tests/unit/test_telegram_settings.py -v`

Expected: 6 passed.

- [ ] **Step 5: Commit the front-end behavior**

```powershell
git add web/static/index.html web/static/app.js tests/unit/test_telegram_frontend_contract.py
git commit -m "feat: auto-save Telegram settings in web UI"
```

---

### Task 4: Verify restart persistence and regression safety

**Files:**
- Verify only; no source change expected.

**Interfaces:**
- Verifies all outputs from Tasks 1-3.

- [ ] **Step 1: Run the complete unit test suite**

Run: `.venv\Scripts\python -X utf8 -m pytest tests/unit -q`

Expected: all unit tests pass with no write to the real `web_settings.json`, because `tests/conftest.py` isolates the settings path.

- [ ] **Step 2: Run syntax checks**

Run: `.venv\Scripts\python -X utf8 -m compileall -q web tests/unit`

Expected: exit code 0.

- [ ] **Step 3: Verify the real settings file remains ignored**

Run: `git check-ignore -v web_settings.json`

Expected: output identifies a `.gitignore` rule for `web_settings.json`.

- [ ] **Step 4: Perform a local restart smoke test without exposing secrets**

With the existing server restarted after implementation:

1. Enter a test Bot Token and Chat ID in the Telegram panel.
2. Wait until the hint says `Telegram 設定已自動儲存。`.
3. Reload the page and confirm all three fields are restored.
4. Stop and restart `run_web.py --port 8765`, reload, and confirm they are still restored.
5. Do not paste the Token into terminal output or commit it.

Expected: values survive both browser reload and process restart.

- [ ] **Step 5: Push verified commits**

Run: `git push origin custom/taiwan-crypto`

Expected: the three Telegram commits are visible on `PurpleLiu/AlphaMaster`, while `web_settings.json` remains untracked.
