from __future__ import annotations

from web.app import (
    TelegramSettingsRequest,
    api_realtime_telegram_get,
    api_realtime_telegram_put,
)


def test_put_then_get_telegram_settings_round_trip() -> None:
    put = api_realtime_telegram_put(TelegramSettingsRequest(
        enabled=True,
        bot_token=" 123456789:AA-test-token ",
        chat_id=" 688220938 ",
    ))
    assert put == {
        "ok": True,
        "enabled": True,
        "active": True,
        "bot_token": "123456789:AA-test-token",
        "chat_id": "688220938",
    }

    get = api_realtime_telegram_get()
    assert get["active"] is True
    assert get["chat_id"] == "688220938"


def test_enabled_with_empty_credentials_is_not_active() -> None:
    response = api_realtime_telegram_put(TelegramSettingsRequest(
        enabled=True,
        bot_token="",
        chat_id="",
    ))
    assert response["enabled"] is True
    assert response["active"] is False


def test_partial_telegram_update_preserves_other_settings() -> None:
    api_realtime_telegram_put(TelegramSettingsRequest(
        enabled=True,
        bot_token="token",
        chat_id="chat",
    ))

    response = api_realtime_telegram_put(TelegramSettingsRequest(enabled=False))

    assert response == {
        "ok": True,
        "enabled": False,
        "active": False,
        "bot_token": "token",
        "chat_id": "chat",
    }
