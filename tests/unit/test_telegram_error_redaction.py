from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

import web.app as app_module
import web.telegram_notify as telegram_notify


client = TestClient(app_module.app)

TOKEN_SENTINEL = "BOT_TOKEN_SENTINEL_7e57b1"
CHAT_ID_SENTINEL = "CHAT_ID_SENTINEL_7e57b1"


def _capture_app_errors(monkeypatch):
    calls: list[str] = []

    def capture(message: str, exc: BaseException | None = None) -> None:
        calls.append(message)
        if exc is not None:
            calls.append(repr(exc))

    monkeypatch.setattr(app_module, "log_error", capture)
    return calls


def _assert_no_credentials(*values: object) -> None:
    text = "\n".join(str(value) for value in values)
    assert TOKEN_SENTINEL not in text
    assert CHAT_ID_SENTINEL not in text


def test_telegram_validation_error_redacts_body_from_response_and_logs(monkeypatch) -> None:
    logged = _capture_app_errors(monkeypatch)

    response = client.put(
        "/api/realtime/telegram",
        json={
            "bot_token": {"raw": TOKEN_SENTINEL},
            "chat_id": [CHAT_ID_SENTINEL],
        },
    )

    assert RequestValidationError in app_module.app.exception_handlers
    assert response.status_code == 422
    assert response.json() == {"detail": "Telegram 設定格式不正確，請檢查欄位格式。"}
    _assert_no_credentials(response.text, logged)


def test_telegram_network_error_redacts_credentials_from_notifier_route_and_logs(monkeypatch) -> None:
    logged = _capture_app_errors(monkeypatch)

    def raise_url_error(*_args, **_kwargs):
        raise URLError(
            f"https://api.telegram.org/bot{TOKEN_SENTINEL}/sendMessage?chat_id={CHAT_ID_SENTINEL}"
        )

    monkeypatch.setattr(telegram_notify.urllib.request, "urlopen", raise_url_error)
    ok, message = telegram_notify.send_text(
        "測試",
        bot_token=TOKEN_SENTINEL,
        chat_id=CHAT_ID_SENTINEL,
    )
    assert ok is False
    _assert_no_credentials(message)

    response = client.post(
        "/api/realtime/telegram/test",
        json={"bot_token": TOKEN_SENTINEL, "chat_id": CHAT_ID_SENTINEL},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Telegram 測試訊息傳送失敗，請檢查網路、Bot Token 與 Chat ID。"
    }
    _assert_no_credentials(response.text, logged)


def test_telegram_http_error_redacts_credentials_from_notifier_route_and_logs(monkeypatch) -> None:
    logged = _capture_app_errors(monkeypatch)

    def raise_http_error(*_args, **_kwargs):
        payload = json.dumps(
            {"description": f"bad token={TOKEN_SENTINEL}; chat={CHAT_ID_SENTINEL}"}
        ).encode("utf-8")
        raise HTTPError(
            f"https://api.telegram.org/bot{TOKEN_SENTINEL}/sendMessage",
            401,
            f"Unauthorized {CHAT_ID_SENTINEL}",
            hdrs=None,
            fp=io.BytesIO(payload),
        )

    monkeypatch.setattr(telegram_notify.urllib.request, "urlopen", raise_http_error)
    ok, message = telegram_notify.send_text(
        "測試",
        bot_token=TOKEN_SENTINEL,
        chat_id=CHAT_ID_SENTINEL,
    )
    assert ok is False
    _assert_no_credentials(message)

    response = client.post(
        "/api/realtime/telegram/test",
        json={"bot_token": TOKEN_SENTINEL, "chat_id": CHAT_ID_SENTINEL},
    )

    assert response.status_code == 400
    _assert_no_credentials(response.text, logged)


def test_telegram_unexpected_error_redacts_credentials_from_response_and_logs(monkeypatch) -> None:
    logged = _capture_app_errors(monkeypatch)

    def raise_sensitive_error(*_args, **_kwargs):
        raise RuntimeError(f"persist {TOKEN_SENTINEL} {CHAT_ID_SENTINEL}")

    monkeypatch.setattr(app_module, "save_settings", raise_sensitive_error)
    safe_client = TestClient(app_module.app, raise_server_exceptions=False)
    response = safe_client.put(
        "/api/realtime/telegram",
        json={"enabled": True},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Telegram 設定處理失敗，請稍後再試。"}
    _assert_no_credentials(response.text, logged)
