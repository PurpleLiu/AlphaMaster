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


def test_partial_telegram_update_preserves_other_settings() -> None:
    client.put("/api/realtime/telegram", json={
        "enabled": True,
        "bot_token": "token",
        "chat_id": "chat",
    })

    response = client.put("/api/realtime/telegram", json={"enabled": False})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "enabled": False,
        "active": False,
        "bot_token": "token",
        "chat_id": "chat",
    }
