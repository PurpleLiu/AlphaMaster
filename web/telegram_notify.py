"""Telegram Bot 通知（信號方向轉折時推送文字，與飛書並行）。

建立 bot：Telegram 找 @BotFather → /newbot → 取得 token。
取得 chat_id：私訊 bot 一則訊息後開
  https://api.telegram.org/bot<token>/getUpdates 看 chat.id。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from web.settings import load_settings, telegram_is_active
from web.feishu_notify import direction_cn, strength_cn

_API_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
_REQUEST_TIMEOUT_S = 10.0


def send_text(
    text: str,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout_s: float = _REQUEST_TIMEOUT_S,
) -> tuple[bool, str]:
    """向 Telegram 發送純文字。返回 (ok, message)。"""
    settings = load_settings()
    token = (bot_token if bot_token is not None else settings.get("telegram_bot_token") or "").strip()
    cid = str(chat_id if chat_id is not None else settings.get("telegram_chat_id") or "").strip()
    if not token:
        return False, "未配置 Bot Token"
    if not cid:
        return False, "未配置 Chat ID"

    payload: dict[str, Any] = {
        "chat_id": cid,
        "text": text,
        "disable_web_page_preview": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _API_TEMPLATE.format(token=token),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("description", "")
        except Exception:
            detail = str(exc)
        return False, f"Telegram HTTP {exc.code}: {detail}"
    except Exception as exc:
        return False, f"Telegram 連線失敗: {exc}"

    if res.get("ok"):
        return True, "ok"
    return False, str(res.get("description", "未知錯誤"))


def notify_direction_flip(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    prev_direction: str,
    new_direction: str,
    strength: float | None = None,
    factor_value: float | None = None,
) -> tuple[bool, str]:
    """信號方向發生轉折時推送提醒（與飛書 notify_direction_flip 同格式）。"""
    settings = load_settings()
    if not telegram_is_active(settings):
        return False, "Telegram 通知未啟用"

    prev_cn = direction_cn(prev_direction)
    new_cn = direction_cn(new_direction)
    grasp = strength_cn(strength, new_direction)
    factor_s = f"{factor_value:+.4f}" if factor_value is not None else "—"

    text = (
        f"【AlphaMaster 信號轉折】\n"
        f"{symbol} · {timeframe}\n"
        f"上次判斷：{prev_cn}\n"
        f"本次判斷：{new_cn}（{grasp}）\n"
        f"策略：{strategy_name}\n"
        f"因子：{factor_s}"
    )
    return send_text(text)
