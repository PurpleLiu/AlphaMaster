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
