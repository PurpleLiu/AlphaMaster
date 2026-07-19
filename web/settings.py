"""Persisted UI settings for the training web console."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "web_settings.json"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
_SETTINGS_LOCK = threading.RLock()

_DEFAULT = {
    "last_data_file": "",
    "last_strategy_file": "",
    "debug_mode": False,
    "ai_provider": "deepseek",
    "ai_api_key": "",
    "ai_base_url": "",
    "ai_model": "",
    # 回測單邊成本（單位 %）：手續費 0.02% + 滑點 0.01% ≈ 常見加密貨幣輕度成本
    "bt_commission_pct": 0.02,
    "bt_slippage_pct": 0.01,
    # 即時分析監控清單：[{source, symbol, timeframe, strategy_file}, ...]
    "realtime_watches": [],
    # 飛書機器人（信號轉折提醒，僅文本）
    "feishu_enabled": False,
    "feishu_webhook_url": "",
    "feishu_secret": "",
    # Telegram 機器人（信號轉折提醒，與飛書並行）
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
}


def telegram_is_active(settings: Mapping[str, Any]) -> bool:
    return bool(settings.get("telegram_enabled")) and bool(
        str(settings.get("telegram_bot_token") or "").strip()
    ) and bool(str(settings.get("telegram_chat_id") or "").strip())


def _write_settings_atomically(settings: Mapping[str, Any]) -> None:
    """Durably replace settings without ever exposing a partial JSON file."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{SETTINGS_PATH.name}.", suffix=".tmp", dir=SETTINGS_PATH.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(settings), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, SETTINGS_PATH)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _as_pct(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0:
        return default
    return v


def _is_ephemeral_data_path(path: str) -> bool:
    """Stale pytest temp parquet paths (not valid training data)."""
    norm = str(path or "").replace("\\", "/").lower()
    if "pytest-of-" not in norm:
        return False
    return (
        "/appdata/local/temp/" in norm
        or norm.startswith("/tmp/")
        or "/temp/" in norm
    )


def _is_production_settings_path() -> bool:
    try:
        return SETTINGS_PATH.resolve() == (PROJECT_ROOT / "web_settings.json").resolve()
    except OSError:
        return False


def _is_usable_data_file(path: str) -> bool:
    p = Path(str(path or "").strip())
    return p.is_file() and p.suffix.lower() == ".parquet"


def _should_replace_last_data_file(path: str) -> bool:
    cur = str(path or "").strip()
    if not cur:
        return True
    if not Path(cur).is_file():
        return True
    return _is_ephemeral_data_path(cur)


def _data_file_from_strategy_json(path: str) -> str | None:
    p = Path(str(path or "").strip())
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    candidate = str(data.get("data_file") or "").strip()
    if _is_usable_data_file(candidate):
        return str(Path(candidate).resolve())
    return None


def _recover_last_data_file(current: dict) -> str:
    cur = str(current.get("last_data_file") or "").strip()
    if cur and not _should_replace_last_data_file(cur):
        return str(Path(cur).resolve())

    for strategy_path in (
        str(current.get("last_strategy_file") or "").strip(),
        *(str(p) for p in sorted(STRATEGIES_DIR.glob("best_*.json")) if p.is_file()),
    ):
        if not strategy_path:
            continue
        candidate = _data_file_from_strategy_json(strategy_path)
        if candidate:
            return candidate
    return cur


def _load_settings_unlocked() -> dict:
    if not SETTINGS_PATH.exists():
        return dict(_DEFAULT)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)
    out = dict(_DEFAULT)
    out.update({k: v for k, v in data.items() if k in _DEFAULT})
    out["debug_mode"] = bool(out.get("debug_mode", False))
    out["last_strategy_file"] = str(out.get("last_strategy_file") or "").strip()
    out["ai_provider"] = str(out.get("ai_provider") or "deepseek").strip().lower()
    if out["ai_provider"] not in ("deepseek", "openclaw", "openclaw_wb"):
        out["ai_provider"] = "deepseek"
    out["ai_api_key"] = str(out.get("ai_api_key") or "").strip()
    out["bt_commission_pct"] = _as_pct(
        out.get("bt_commission_pct"), _DEFAULT["bt_commission_pct"]
    )
    out["bt_slippage_pct"] = _as_pct(
        out.get("bt_slippage_pct"), _DEFAULT["bt_slippage_pct"]
    )
    watches = out.get("realtime_watches")
    if not isinstance(watches, list):
        watches = []
    cleaned = []
    for w in watches:
        if not isinstance(w, dict):
            continue
        src = str(w.get("source") or "").strip()
        sym = str(w.get("symbol") or "").strip()
        tf = str(w.get("timeframe") or "").strip()
        sf = str(w.get("strategy_file") or "").strip()
        if src and sym and tf and sf:
            cleaned.append(
                {"source": src, "symbol": sym, "timeframe": tf, "strategy_file": sf}
            )
    out["realtime_watches"] = cleaned
    out["feishu_enabled"] = bool(out.get("feishu_enabled", False))
    out["feishu_webhook_url"] = str(out.get("feishu_webhook_url") or "").strip()
    out["feishu_secret"] = str(out.get("feishu_secret") or "").strip()
    out["telegram_enabled"] = bool(out.get("telegram_enabled", False))
    out["telegram_bot_token"] = str(out.get("telegram_bot_token") or "").strip()
    out["telegram_chat_id"] = str(out.get("telegram_chat_id") or "").strip()
    if _is_production_settings_path():
        recovered = _recover_last_data_file(out)
        if recovered != out.get("last_data_file"):
            out["last_data_file"] = recovered
            if recovered:
                _write_settings_atomically(out)
    elif _is_ephemeral_data_path(out.get("last_data_file", "")):
        # Tests and alternate settings files may still exercise recovery from
        # their explicitly selected strategy, but must never scan production
        # strategies or replace an ordinary caller-provided path.
        recovered = _data_file_from_strategy_json(out.get("last_strategy_file", ""))
        if recovered:
            out["last_data_file"] = recovered
    return out


def load_settings() -> dict:
    with _SETTINGS_LOCK:
        return _load_settings_unlocked()


def _save_settings_unlocked(data: dict) -> dict:
    current = _load_settings_unlocked()
    if "last_data_file" in data:
        path = str(data["last_data_file"] or "").strip()
        if (
            path
            and _is_ephemeral_data_path(path)
            and _is_production_settings_path()
        ):
            data = {k: v for k, v in data.items() if k != "last_data_file"}
        else:
            current["last_data_file"] = path
    if "last_strategy_file" in data:
        current["last_strategy_file"] = str(data["last_strategy_file"] or "").strip()
    if "debug_mode" in data:
        current["debug_mode"] = bool(data["debug_mode"])
    if "ai_provider" in data:
        provider = str(data["ai_provider"] or "deepseek").strip().lower()
        current["ai_provider"] = (
            provider if provider in ("deepseek", "openclaw", "openclaw_wb") else "deepseek"
        )
    if "ai_api_key" in data:
        current["ai_api_key"] = str(data["ai_api_key"] or "").strip()
    if "bt_commission_pct" in data:
        current["bt_commission_pct"] = _as_pct(
            data["bt_commission_pct"], _DEFAULT["bt_commission_pct"]
        )
    if "bt_slippage_pct" in data:
        current["bt_slippage_pct"] = _as_pct(
            data["bt_slippage_pct"], _DEFAULT["bt_slippage_pct"]
        )
    if "realtime_watches" in data:
        watches = data["realtime_watches"]
        if not isinstance(watches, list):
            watches = []
        cleaned = []
        for w in watches:
            if not isinstance(w, dict):
                continue
            src = str(w.get("source") or "").strip()
            sym = str(w.get("symbol") or "").strip()
            tf = str(w.get("timeframe") or "").strip()
            sf = str(w.get("strategy_file") or "").strip()
            if src and sym and tf and sf:
                cleaned.append(
                    {
                        "source": src,
                        "symbol": sym,
                        "timeframe": tf,
                        "strategy_file": sf,
                    }
                )
        current["realtime_watches"] = cleaned
    if "feishu_enabled" in data:
        current["feishu_enabled"] = bool(data["feishu_enabled"])
    if "feishu_webhook_url" in data:
        current["feishu_webhook_url"] = str(data["feishu_webhook_url"] or "").strip()
    if "feishu_secret" in data:
        current["feishu_secret"] = str(data["feishu_secret"] or "").strip()
    if "telegram_enabled" in data:
        current["telegram_enabled"] = bool(data["telegram_enabled"])
    if "telegram_bot_token" in data:
        current["telegram_bot_token"] = str(data["telegram_bot_token"] or "").strip()
    if "telegram_chat_id" in data:
        current["telegram_chat_id"] = str(data["telegram_chat_id"] or "").strip()
    _write_settings_atomically(current)
    return current


def save_settings(data: dict) -> dict:
    """Serialize the entire read-merge-write operation for every settings API."""
    with _SETTINGS_LOCK:
        return _save_settings_unlocked(data)
