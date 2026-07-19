"""數據源工廠 + 單例註冊（復用連接）。"""
from __future__ import annotations

import threading

from web.data_sources.base import DataSource

# 前端下拉可見的數據源（okx / tongdaxin 實現保留，暫不展示）
SOURCE_KINDS: tuple[tuple[str, str], ...] = (
    ("mt5", "MT5"),
    ("tradingview", "TradingView"),
)

_INSTANCES: dict[str, DataSource] = {}
_LOCK = threading.Lock()


def _build(kind: str) -> DataSource:
    if kind == "mt5":
        from web.data_sources.mt5_source import MT5Source
        return MT5Source()
    if kind == "tradingview":
        from web.data_sources.tradingview_source import TradingViewSource
        return TradingViewSource()
    if kind == "okx":
        from web.data_sources.okx_source import OKXSource
        return OKXSource()
    if kind == "tongdaxin":
        from web.data_sources.tongdaxin_source import TongdaxinSource
        return TongdaxinSource()
    raise ValueError(f"未知數據源: {kind}")


def get_source(kind: str) -> DataSource:
    """返回該 kind 的單例數據源（懶創建，復用連接）。"""
    with _LOCK:
        inst = _INSTANCES.get(kind)
        if inst is None:
            inst = _build(kind)
            _INSTANCES[kind] = inst
        return inst


def list_sources() -> list[dict]:
    """列出所有數據源及其可用狀態（供前端灰顯/引導）。"""
    out = []
    for kind, label in SOURCE_KINDS:
        try:
            src = get_source(kind)
            ok, hint = src.available()
            tfs = src.supported_timeframes()
            presets = src.preset_symbols()
        except Exception as exc:  # noqa: BLE001
            ok, hint, tfs, presets = False, str(exc), [], []
        out.append(
            {
                "id": kind,
                "label": label,
                "available": ok,
                "hint": hint,
                "timeframes": tfs,
                "presets": presets,
            }
        )
    return out
