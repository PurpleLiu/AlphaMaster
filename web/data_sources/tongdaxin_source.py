"""通達信數據源（pytdx，免費行情伺服器，A 股 / 指數）。"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 通達信行情伺服器（多個備選）
_SERVERS = [
    ("115.238.90.165", 7709),
    ("180.153.18.170", 7709),
    ("119.147.212.81", 7709),
    ("14.17.75.71", 7709),
    ("59.173.18.77", 7709),
]

# 項目週期 -> pytdx category
_CAT = {
    "1m": 8,
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "1h": 3,
    "1d": 9,
    "1w": 5,
    "1M": 6,
}

_PRESETS = ["600519", "000001", "300750", "601318", "000858", "sh000001", "sz399006"]


def _parse_market(code: str) -> tuple[int, str]:
    """返回 (market, pure_code)。1=上海, 0=深圳。"""
    c = code.strip().upper()
    if c.startswith("SH"):
        return 1, c[2:]
    if c.startswith("SZ"):
        return 0, c[2:]
    if c[:1] in ("6", "5", "9") or c.startswith("11") or c.startswith("13"):
        return 1, c
    return 0, c


_CST = timezone(timedelta(hours=8))  # 通達信返回的 datetime 為北京時間（UTC+8）


def _is_index(market: int, code: str) -> bool:
    """判斷是否為指數代碼。

    上證指數系列：market=1 且 code 以 000 開頭（如 000001 上證指數、000300 滬深300）。
    深證指數系列：market=0 且 code 以 399 開頭（如 399001 深證成指、399006 創業板指）。
    注意：深圳市場 000xxx 是股票（如 000001 平安銀行），只有 399xxx 才是深證指數，
    因此不能用純 code 前綴判斷，必須結合 market。
    """
    if market == 1 and code.startswith("000"):
        return True
    if market == 0 and code.startswith("399"):
        return True
    return False


def _parse_dt(s: str) -> int:
    """通達信 datetime（北京時間，如 '2026-07-15 15:00'）-> 收盤時刻的 Unix 秒。

    返回值語義為「該 bar 的收盤時刻（UTC 秒）」；drop_forming / _ensure_closed_bars
    據此判斷是否已收盤（ts > now 即仍在形成中）。
    """
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(str(s), fmt).replace(tzinfo=_CST).timestamp())
        except ValueError:
            continue
    return 0


def _looks_corrupted(raw) -> bool:
    """檢測返回數據是否亂碼（datetime 年份異常）。

    典型場景：對指數誤用 get_security_bars，第 2 條起 datetime 損壞（如 7772-67-85）；
    或純數字指數代碼未帶 sh/sz 前綴（如 000300）導致市場誤判後取到異常數據。
    """
    if not raw:
        return False
    bad = 0
    for r in raw:
        s = str(r.get("datetime", ""))
        y = int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else -1
        if not (1990 <= y <= 2035):
            bad += 1
    return bad > 0 and bad >= len(raw) * 0.3


class TongdaxinSource(DataSource):
    kind = "tongdaxin"
    label = "通達信"

    def __init__(self) -> None:
        self._api = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import pytdx  # noqa: F401
        except ImportError:
            return (False, "未安裝 pytdx：pip install pytdx")
        return (True, "免費行情伺服器 · A 股 / 指數")

    def supported_timeframes(self) -> list[str]:
        return list(_CAT.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        if self._api is not None:
            return
        try:
            from pytdx.hq import TdxHq_API
        except ImportError as exc:
            raise DataSourceUnavailable("未安裝 pytdx") from exc
        api = TdxHq_API()
        for host, port in _SERVERS:
            try:
                if api.connect(host, port):
                    self._api = api
                    return
            except Exception:
                continue
        raise DataSourceUnavailable("通達信所有行情伺服器連接失敗")

    def disconnect(self) -> None:
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
        self._api = None

    def _fetch_raw(self, cat: int, market: int, code: str, want: int, is_index: bool):
        """指數走 get_index_bars，股票走 get_security_bars。

        通達信協議規定指數必須用 get_index_bars；若對指數用 get_security_bars，
        返回數據從第 2 條起 datetime 會損壞（年份變成 7772、228200 等亂碼）。
        """
        if is_index:
            return self._api.get_index_bars(cat, market, code, 0, want)
        return self._api.get_security_bars(cat, market, code, 0, want)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        """拉取 K 線。

        注意：返回的是不復權數據（pytdx 限制），歷史含除權跳空；volume 單位為
        「手」（1手=100股），與 OKX/MT5 的 volume 量綱不同，跨源不可比。
        """
        if timeframe not in _CAT:
            raise DataSourceUnavailable(f"通達信不支持週期 {timeframe}")
        market, code = _parse_market(symbol)
        cat = _CAT[timeframe]
        want = min(max(n + 2, 20), 800)  # 單次上限 800
        is_index = _is_index(market, code)

        with self._lock:
            self.connect()
            try:
                raw = self._fetch_raw(cat, market, code, want, is_index)
            except Exception:
                # 連接可能失效，重連一次
                self._api = None
                self.connect()
                raw = self._fetch_raw(cat, market, code, want, is_index)

        if not raw:
            raise DataSourceUnavailable(
                f"通達信無數據：{symbol}。請確認代碼正確；指數需帶 sh/sz 前綴（如 sh000001）。"
            )
        if _looks_corrupted(raw):
            # 數據亂碼通常意味著介面選錯（指數誤用股票介面）或代碼/市場不匹配。
            # 純數字指數代碼（如 000300）未帶 sh/sz 前綴時會誤判市場，此處給出明確提示。
            raise DataSourceUnavailable(
                f"通達信返回數據異常：{symbol}。若為指數請使用 sh/sz 前綴"
                f"（如 sh000001、sz399006），股票代碼請確認無誤。"
            )

        bars: list[Bar] = []
        for r in raw:
            bars.append(
                Bar(
                    ts=_parse_dt(r.get("datetime", "")),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("vol", 0.0) or 0.0),  # 單位：手（1手=100股）
                )
            )
        bars.sort(key=lambda b: b.ts)  # 保證升序
        # 剔除尚未收盤的 bar：通達信 datetime 為收盤時刻（北京時間），_parse_dt
        # 返回的 ts 即收盤時刻的 UTC 秒；ts > now 說明該 bar 仍在形成中。
        # （不再盲刪最後一條，以免盤後把當天已收盤 bar 誤刪，導致 last_bar 滯後一天。）
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()
        return bars[-n:]
