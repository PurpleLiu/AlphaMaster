from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

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


def test_save_settings_keeps_previous_file_when_atomic_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {"ai_model": "model-router", "telegram_bot_token": "existing"}
    settings_mod.SETTINGS_PATH.write_text(json.dumps(original), encoding="utf-8")

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(settings_mod.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        save_settings({"ai_model": "replacement"})

    assert json.loads(settings_mod.SETTINGS_PATH.read_text(encoding="utf-8")) == original
    assert not list(settings_mod.SETTINGS_PATH.parent.glob("*.tmp"))


def test_save_settings_serializes_concurrent_disjoint_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_mod.SETTINGS_PATH.write_text(json.dumps({"debug_mode": False}), encoding="utf-8")
    target = settings_mod.SETTINGS_PATH
    original_write = settings_mod._write_settings_atomically
    first_write_waiting = Event()
    release_first_write = Event()
    write_count = 0
    guard = Lock()

    def delayed_write(settings):
        nonlocal write_count
        with guard:
            write_count += 1
            is_first = write_count == 1
        if is_first:
            first_write_waiting.set()
            assert release_first_write.wait(timeout=3)
        return original_write(settings)

    monkeypatch.setattr(settings_mod, "_write_settings_atomically", delayed_write)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(save_settings, {"debug_mode": True})
        assert first_write_waiting.wait(timeout=3)
        second = pool.submit(save_settings, {"telegram_enabled": True})
        release_first_write.set()
        first.result(timeout=5)
        second.result(timeout=5)

    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["debug_mode"] is True
    assert saved["telegram_enabled"] is True
