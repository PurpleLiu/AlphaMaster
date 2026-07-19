"""OKX 數據源（公開 REST，USDT 永續 / 現貨）。"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

OKX_BASE = "https://www.okx.com"

# 項目週期 -> OKX bar
_TF = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
    "1M": "1M",
}

_PRESETS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "LINKUSDT",
]


def _normalize_inst_id(symbol: str) -> str:
    """BTCUSDT / BTC-USDT / BTC-USDT-SWAP -> BTC-USDT-SWAP（優先永續）。"""
    s = (symbol or "").strip().upper().replace("/", "-").replace("_", "-")
    if not s:
        raise DataSourceUnavailable("請填寫品種")
    if s.endswith("-SWAP"):
        return s
    if s.count("-") >= 1:
        # BTC-USDT / ETH-USDT-SWAP
        parts = s.split("-")
        if len(parts) >= 2 and parts[1] in ("USDT", "USD", "USDC"):
            return f"{parts[0]}-{parts[1]}-SWAP"
        return s
    # BTCUSDT / BTCUSD
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            base = s[: -len(quote)]
            return f"{base}-{quote}-SWAP"
    raise DataSourceUnavailable(f"無法識別 OKX 品種：{symbol}（範例 BTCUSDT）")


def _okx_get(path: str, params: dict[str, str], retries: int = 3) -> list:
    query = urllib.parse.urlencode(params)
    url = f"{OKX_BASE}{path}?{query}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaMaster/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("code") != "0":
                raise RuntimeError(f"OKX API {body.get('code')}: {body.get('msg')}")
            return body.get("data") or []
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(min(8, 2**attempt))
    raise DataSourceUnavailable(f"OKX 請求失敗: {last_err}")


class OKXSource(DataSource):
    kind = "okx"
    label = "OKX"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        return (True, "公開行情 · USDT 永續（範例 BTCUSDT）")

    def supported_timeframes(self) -> list[str]:
        return list(_TF.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _TF:
            raise DataSourceUnavailable(f"OKX 不支持週期 {timeframe}")
        inst_id = _normalize_inst_id(symbol)
        bar = _TF[timeframe]
        want = min(max(n + 2, 20), 300)

        with self._lock:
            try:
                raw = _okx_get(
                    "/api/v5/market/candles",
                    {"instId": inst_id, "bar": bar, "limit": str(want)},
                )
            except DataSourceUnavailable:
                raw = []
            if (not raw) and inst_id.endswith("-SWAP"):
                spot = inst_id[: -len("-SWAP")]
                raw = _okx_get(
                    "/api/v5/market/candles",
                    {"instId": spot, "bar": bar, "limit": str(want)},
                )

        if not raw:
            raise DataSourceUnavailable(f"OKX 無數據：{symbol}")

        bars: list[Bar] = []
        for item in raw:
            # [ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            bars.append(
                Bar(
                    ts=int(int(item[0]) // 1000),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5] or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts)
        if drop_forming and len(bars) > 1:
            newest = raw[0]  # API 返回降序
            # confirm==0 未收盤；缺欄位時保守剔除最新一根
            if len(newest) <= 8 or str(newest[8]) == "0":
                bars = bars[:-1]
        return bars[-n:]
