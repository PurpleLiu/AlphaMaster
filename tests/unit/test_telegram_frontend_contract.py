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
    assert "function createSerialTaskQueue()" in source
    assert "let rtTelegramSaveQueue = createSerialTaskQueue();" in source
    assert "rtTelegramToken\")) $(\"rtTelegramToken\").value = \"\";" in source
    assert "data.bot_token" not in source
    assert "data.chat_id ||" not in source
    assert "已儲存，留空保留" in source
    assert 'placeholder="已儲存時留空即可保留；輸入新 Token 才會更新"' in html
    assert 'id="rtTelegramSaveBtn">儲存設定</button>' in html
