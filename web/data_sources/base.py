"""即時行情數據源抽象層。

統一介面，供 realtime_manager 拉取 K 線並轉換為 AlphaMaster 特徵引擎所需的
raw_dict（torch 張量 [1, T]，升序=最舊在前）。

參考 PA_Agent 的 DataSource 設計，但做成自包含、可選依賴優雅降級。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# 項目統一的週期字串（各源在內部映射到自己的常量）
CANON_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")


class DataSourceError(Exception):
    """數據源通用錯誤。"""


class DataSourceUnavailable(DataSourceError):
    """依賴缺失或無法連接（前端據此灰顯該源）。"""


@dataclass
class Bar:
    """單根 K 線（ts = 開盤時間的 Unix 秒）。"""
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataSource(ABC):
    """K 線數據源統一介面。"""

    kind: str = ""
    label: str = ""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """返回 (是否可用, 提示文案)。用於前端灰顯與安裝引導。"""

    @abstractmethod
    def supported_timeframes(self) -> list[str]:
        """返回該源支持的週期字串列表（CANON_TIMEFRAMES 子集）。"""

    @abstractmethod
    def preset_symbols(self) -> list[str]:
        """返回預設/常用品種列表（不阻塞網路）。"""

    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        """拉取最近 n 根已收盤 K 線，升序（最舊在前）。

        drop_forming=True 時剔除當前正在形成的 bar，使最後一根為「最後已收盤 bar」。
        """

    def connect(self) -> None:  # noqa: B027 - 可選
        """建立/復用連接（可選）。"""

    def disconnect(self) -> None:  # noqa: B027 - 可選
        """斷開連接（可選）。"""


def bars_to_raw_dict(bars: list[Bar]):
    """將升序 Bar 列錶轉換為 AlphaMaster 特徵引擎所需的 raw_dict（torch [1, T]）。"""
    import torch

    if not bars:
        raise DataSourceError("空 K 線序列")

    def col(vals: list[float]):
        return torch.tensor([vals], dtype=torch.float32)

    return {
        "open": col([b.open for b in bars]),
        "high": col([b.high for b in bars]),
        "low": col([b.low for b in bars]),
        "close": col([b.close for b in bars]),
        "volume": col([max(b.volume, 0.0) for b in bars]),
        "time": col([float(b.ts) for b in bars]),
    }
