"""
strategy_manager/portfolio.py — MT5 倉位管理器

管理 MT5 倉位狀態，支持 JSON 持久化和 MT5 即時同步。
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict

from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None

try:
    from config import Config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    # 測試環境回退預設值
    class Config:
        PORTFOLIO_FILE = "portfolio_state.json"


@dataclass
class Position:
    """MT5 倉位數據結構。"""
    symbol: str
    ticket: int
    entry_price: float
    entry_time: float
    lot_size: float
    direction: str          # "BUY" | "SELL"
    highest_price: float    # 多頭追蹤止損用（最高價）
    lowest_price: float     # 空頭追蹤止損用（最低價）
    is_partial_closed: bool


class MT5PortfolioManager:
    """MT5 倉位管理器。

    負責記錄、更新、持久化倉位狀態，並與 MT5 終端即時同步。
    """

    def __init__(self) -> None:
        self.positions: Dict[str, Position] = {}
        self.state_file: str = Config.PORTFOLIO_FILE
        self.load_state()

    # ─────────────────────────────────────────────────────────────
    # 倉位增刪改查
    # ─────────────────────────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        ticket: int,
        price: float,
        lot: float,
        direction: str,
    ) -> None:
        """記錄一個新開倉位。

        Args:
            symbol:    交易品種
            ticket:    MT5 order ticket（由 mt5.order_send() 返回）
            price:     入場價格
            lot:       手數
            direction: "BUY" 或 "SELL"
        """
        pos = Position(
            symbol=symbol,
            ticket=ticket,
            entry_price=price,
            entry_time=time.time(),
            lot_size=lot,
            direction=direction,
            highest_price=price,
            lowest_price=price,
            is_partial_closed=False,
        )
        self.positions[symbol] = pos
        self.save_state()
        logger.info(f"[Portfolio] Position added: {symbol} {direction} lot={lot} @ {price} ticket={ticket}")

    def close_position(self, symbol: str) -> None:
        """從本地狀態移除倉位（不發出 MT5 訂單，僅清除記錄）。

        Args:
            symbol: 要關閉的品種
        """
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self.save_state()
            logger.info(f"[Portfolio] Position closed: {symbol} ticket={pos.ticket}")
        else:
            logger.warning(f"[Portfolio] close_position: {symbol} not found in local state")

    def get_direction(self, symbol: str) -> int:
        """返回品種當前持倉方向的整數表示。

        Returns:
            +1 多頭 / -1 空頭 / 0 空倉
        """
        if symbol not in self.positions:
            return 0
        d = self.positions[symbol].direction
        return 1 if d == "BUY" else -1

    def update_price(self, symbol: str, price: float) -> None:
        """更新當前價格，分別追蹤多頭最高價和空頭最低價。"""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        changed = False
        if pos.direction == "BUY" and price > pos.highest_price:
            pos.highest_price = price
            changed = True
        elif pos.direction == "SELL" and price < pos.lowest_price:
            pos.lowest_price = price
            changed = True
        if changed:
            self.save_state()

    def get_open_count(self) -> int:
        """返回當前持倉數量。"""
        return len(self.positions)

    # ─────────────────────────────────────────────────────────────
    # MT5 同步
    # ─────────────────────────────────────────────────────────────

    def sync_from_mt5(self) -> None:
        """與 MT5 終端同步倉位狀態。

        1. 調用 mt5.positions_get() 獲取當前所有持倉。
        2. 將本地記錄中已不在 MT5 的倉位移除（外部平倉）。
        3. 將 MT5 中有但本地沒有的倉位補錄（漏記情況）。
        4. 同步 direction（以 MT5 為準）。
        """
        if not _MT5_AVAILABLE or mt5 is None:
            logger.warning("[Portfolio] MT5 not available, skipping sync")
            return

        live_positions = mt5.positions_get()
        if live_positions is None:
            logger.warning(f"[Portfolio] mt5.positions_get() failed: {mt5.last_error()}")
            return

        allowed_symbols = set(getattr(Config, "SYMBOLS", []) or [])
        excluded_symbols = set(getattr(Config, "EXCLUDED_TRADE_SYMBOLS", []) or [])

        # 以 symbol 為 key 建立 MT5 持倉索引，僅同步當前自動交易品種。
        # 相容 mock：若 symbol 屬性是字串才使用；否則降級為按 ticket 匹配
        live_by_symbol: dict[str, object] = {}
        live_tickets: set[int] = set()
        for p in live_positions:
            sym    = getattr(p, "symbol", None)
            ticket = getattr(p, "ticket", None)
            if isinstance(sym, str):
                if sym in excluded_symbols:
                    continue
                if allowed_symbols and sym not in allowed_symbols:
                    continue
            if isinstance(sym, str):
                live_by_symbol[sym] = p
            if isinstance(ticket, int):
                live_tickets.add(ticket)

        if live_by_symbol:
            # 新邏輯：按 symbol 對帳（symbol 屬性為字串時）
            to_remove = [s for s in self.positions if s not in live_by_symbol]
            for s in to_remove:
                pos = self.positions.pop(s)
                logger.info(f"[Portfolio] Externally closed, removed: {s} ticket={pos.ticket}")

            # 同步 direction 並補錄漏記倉位
            for sym, p in live_by_symbol.items():
                direction = "BUY" if getattr(p, "type", 0) == 0 else "SELL"
                if sym in self.positions:
                    self.positions[sym].direction = direction
                else:
                    price = float(getattr(p, "price_open", 0.0))
                    self.positions[sym] = Position(
                        symbol=sym,
                        ticket=getattr(p, "ticket", 0),
                        entry_price=price,
                        entry_time=float(getattr(p, "time", time.time())),
                        lot_size=float(getattr(p, "volume", 0.01)),
                        direction=direction,
                        highest_price=price,
                        lowest_price=price,
                        is_partial_closed=False,
                    )
                    logger.info(f"[Portfolio] 補錄MT5持倉: {sym} {direction}")
        else:
            # 降級邏輯：只有 ticket 可用時，按 ticket 移除已不存在的倉位
            to_remove = [
                s for s, pos in self.positions.items()
                if pos.ticket not in live_tickets
            ]
            for s in to_remove:
                pos = self.positions.pop(s)
                logger.info(f"[Portfolio] Externally closed (by ticket), removed: "
                            f"{s} ticket={pos.ticket}")

        if to_remove:
            self.save_state()

    # ─────────────────────────────────────────────────────────────
    # JSON 持久化
    # ─────────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """將當前倉位狀態保存到 JSON 文件（Config.PORTFOLIO_FILE）。"""
        data = {symbol: asdict(pos) for symbol, pos in self.positions.items()}
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"[Portfolio] State saved to {self.state_file} ({len(data)} positions)")
        except OSError as e:
            logger.error(f"[Portfolio] Failed to save state: {e}")

    def load_state(self) -> None:
        """從 JSON 文件恢復倉位狀態。

        文件不存在時靜默初始化為空倉位集合。
        JSON 欄位不匹配時記錄 WARNING 並跳過該條目。
        """
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data: dict = json.load(f)

            loaded = 0
            for symbol, fields in data.items():
                try:
                    self.positions[symbol] = Position(**fields)
                    loaded += 1
                except (TypeError, KeyError) as e:
                    logger.warning(
                        f"[Portfolio] Skipping malformed position '{symbol}': {e}"
                    )

            logger.info(
                f"[Portfolio] Loaded {loaded} position(s) from {self.state_file}"
            )

        except FileNotFoundError:
            logger.info(
                f"[Portfolio] No state file found at {self.state_file}, starting fresh"
            )
            self.positions = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[Portfolio] Failed to load state: {e}")
            self.positions = {}
