from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app


client = TestClient(app)


def test_put_then_get_telegram_settings_round_trip() -> None:
    raw_token = "123456789:AA-test-token"
    raw_chat_id = "688220938"
    put = client.put("/api/realtime/telegram", json={
        "enabled": True,
        "bot_token": f" {raw_token} ",
        "chat_id": f" {raw_chat_id} ",
    })
    assert put.status_code == 200
    assert put.json() == {
        "ok": True,
        "enabled": True,
        "active": True,
        "token_configured": True,
        "token_hint": "已儲存，留空保留",
        "chat_id_configured": True,
    }
    assert raw_token not in put.text
    assert raw_chat_id not in put.text

    get = client.get("/api/realtime/telegram")
    assert get.status_code == 200
    assert get.json() == {
        "enabled": True,
        "active": True,
        "token_configured": True,
        "token_hint": "已儲存，留空保留",
        "chat_id_configured": True,
    }
    assert raw_token not in get.text
    assert raw_chat_id not in get.text

    for response in (
        client.get("/api/settings"),
        client.put("/api/settings", json={"debug_mode": True}),
    ):
        assert response.status_code == 200
        assert raw_token not in response.text
        assert raw_chat_id not in response.text


def test_blank_credentials_preserve_stored_values_until_explicitly_cleared() -> None:
    client.put("/api/realtime/telegram", json={
        "enabled": True,
        "bot_token": "token",
        "chat_id": "chat",
    })

    response = client.put("/api/realtime/telegram", json={
        "enabled": False,
        "bot_token": " ",
        "chat_id": " ",
    })
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["active"] is False
    assert response.json()["token_configured"] is True
    assert response.json()["chat_id_configured"] is True

    cleared = client.put("/api/realtime/telegram", json={
        "clear_token": True,
        "clear_chat_id": True,
    })
    assert cleared.status_code == 200
    assert cleared.json()["token_configured"] is False
    assert cleared.json()["chat_id_configured"] is False


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
        "token_configured": True,
        "token_hint": "已儲存，留空保留",
        "chat_id_configured": True,
    }


def test_telegram_cors_allows_only_local_origins_without_exposing_credentials() -> None:
    client.put("/api/realtime/telegram", json={
        "bot_token": "cors-test-token",
        "chat_id": "cors-test-chat",
    })

    allowed = client.options(
        "/api/realtime/telegram",
        headers={
            "Origin": "http://localhost:8765",
            "Access-Control-Request-Method": "PUT",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8765"

    rejected = client.options(
        "/api/realtime/telegram",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "PUT",
        },
    )
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers
    assert "cors-test-token" not in rejected.text
    assert "cors-test-chat" not in rejected.text

    cross_origin_get = client.get(
        "/api/realtime/telegram",
        headers={"Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in cross_origin_get.headers
    assert "cors-test-token" not in cross_origin_get.text
    assert "cors-test-chat" not in cross_origin_get.text
