"""
strategy_manager/risk.py — MT5 風控引擎

實現 MT5RiskEngine.calculate_lot()：
  - 基於帳戶淨值和 RISK_PER_TRADE 計算手數
  - 通過 mt5.symbol_info() 獲取 pip_value（trade_tick_value）、
    volume_step、volume_min、volume_max
  - lot = (equity * RISK_PER_TRADE) / (stop_pips * pip_value)
  - 捨入到 volume_step，clamp 到 [volume_min, volume_max]
  - 保證金不足時返回 0.0 並記錄 WARNING
"""

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

import math
from loguru import logger

try:
    from config import Config
except ImportError:
    # 測試環境回退
    class Config:  # type: ignore
        RISK_PER_TRADE = 0.01


class MT5RiskEngine:
    """MT5 手數計算引擎（Requirements 9.1–9.6）。

    所有方法均為同步介面，無 asyncio。
    """

    def __init__(self, risk_per_trade: float | None = None):
        """初始化風控引擎。

        Args:
            risk_per_trade: 每筆風險比例，預設使用 Config.RISK_PER_TRADE (0.01)。
        """
        self.risk_per_trade: float = (
            risk_per_trade if risk_per_trade is not None else Config.RISK_PER_TRADE
        )

    # ------------------------------------------------------------------
    # 公共介面
    # ------------------------------------------------------------------

    def calculate_lot(
        self,
        symbol: str,
        equity: float,
        stop_pips: float,
    ) -> float:
        """根據帳戶淨值和止損 pips 計算手數。

        公式：
            desired_lot = (equity * RISK_PER_TRADE) / (stop_pips * pip_value)
            lot = round(desired_lot / volume_step) * volume_step
            lot = clamp(lot, volume_min, volume_max)

        若保證金不足，返回 0.0 並記錄 WARNING。

        Args:
            symbol:     MT5 品種，例如 "XAUUSD"
            equity:     帳戶淨值（帳戶貨幣）
            stop_pips:  止損 pips 數（> 0）

        Returns:
            有效手數，或 0.0（保證金不足 / 參數無效）。

        Requirements:
            9.1 基於 equity 和 RISK_PER_TRADE 計算手數
            9.2 使用 mt5.symbol_info() 的 trade_tick_value 作為 pip_value
            9.3 捨入到 volume_step
            9.4 clamp 到 [volume_min, volume_max]
            9.5 不包含任何 Honeypot / DEX 流動性檢查
            9.6 保證金不足返回 0.0 並記錄 WARNING；保證金檢查在手數計算之後
        """
        # ── 參數防禦 ──────────────────────────────────────────────────
        if stop_pips <= 0:
            logger.warning(f"[RiskEngine] Invalid stop_pips={stop_pips} for {symbol}")
            return 0.0
        if equity <= 0:
            logger.warning(f"[RiskEngine] Invalid equity={equity} for {symbol}")
            return 0.0

        # ── 9.2 從 MT5 獲取品種規格 ───────────────────────────────────
        symbol_info = self._get_symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[RiskEngine] Cannot get symbol_info for {symbol}")
            return 0.0

        pip_value: float = symbol_info.trade_tick_value
        volume_step: float = symbol_info.volume_step
        volume_min: float = symbol_info.volume_min
        volume_max: float = symbol_info.volume_max

        if pip_value <= 0 or volume_step <= 0:
            logger.warning(
                f"[RiskEngine] Invalid symbol specs for {symbol}: "
                f"pip_value={pip_value}, volume_step={volume_step}"
            )
            return 0.0

        # ── 9.1 計算目標手數 ──────────────────────────────────────────
        desired_lot: float = (equity * self.risk_per_trade) / (stop_pips * pip_value)

        # ── 9.3 捨入到 volume_step ────────────────────────────────────
        lot: float = round(desired_lot / volume_step) * volume_step

        # ── 9.4 clamp 到 [volume_min, volume_max] ────────────────────
        lot = max(volume_min, min(lot, volume_max))

        # ── 9.6 保證金檢查（在手數計算之後）─────────────────────────
        if not self._has_sufficient_margin(symbol, lot, pip_value, stop_pips):
            return 0.0

        return lot

    # ------------------------------------------------------------------
    # ATR 波動率目標倉位（推薦方法）
    # ------------------------------------------------------------------

    def calculate_lot_by_atr(
        self,
        symbol:         str,
        equity:         float,
        atr_price:      float,
        target_risk_pct: float | None = None,
        max_lot:        float         = 0.1,
    ) -> float:
        """基於 ATR 的波動率目標倉位計算。

        每筆交易的預期盈虧金額 = target_risk_pct × equity，
        手數 = 目標風險金額 / (ATR × 合約價值/手)

        這樣不同品種（黃金/美日/納指/標普）下單後，
        每 1 個 ATR 波動對應的盈虧金額是相同的，資金暴露均衡。

        Args:
            symbol:          MT5 品種名
            equity:          帳戶淨值（帳戶貨幣）
            atr_price:       品種價格單位的 ATR 值（已乘以合約乘數前）
            target_risk_pct: 目標風險比例，None 時用 Config.RISK_PER_TRADE
            max_lot:         手數硬性上限

        Returns:
            有效手數（已捨入，已 clamp），或 0.01（最小手數回退）
        """
        if target_risk_pct is None:
            target_risk_pct = self.risk_per_trade

        if equity <= 0 or atr_price <= 0:
            logger.warning(f"[RiskEngine.atr] Invalid equity={equity} or atr={atr_price}")
            return 0.0

        symbol_info = self._get_symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[RiskEngine.atr] No symbol_info for {symbol}")
            return 0.01

        # 合約價值（每手）= contract_size × tick_value / tick_size × 當前價格
        # 簡化：使用 trade_tick_value / trade_tick_size 得到每價格單位每手的盈虧
        tick_val  = symbol_info.trade_tick_value   # 每 tick 每手的盈虧（帳戶貨幣）
        tick_size = symbol_info.trade_tick_size    # 每 tick 的價格變化
        if tick_val <= 0 or tick_size <= 0:
            logger.warning(f"[RiskEngine.atr] Invalid tick data for {symbol}")
            return symbol_info.volume_min

        # 每價格單位每手的盈虧 = tick_val / tick_size
        value_per_unit = tick_val / tick_size       # $/price_unit/lot

        # 目標風險金額 = equity × target_risk_pct
        target_risk_usd = equity * target_risk_pct

        # 手數 = 目標風險 / (ATR × 每單位價值/手)
        desired_lot = target_risk_usd / (atr_price * value_per_unit)

        # 固定手數品種（貴金屬等），跳過 ATR 計算
        fixed_map = getattr(Config, "FIXED_LOT_BY_SYMBOL", {}) or {}
        sym_key = symbol.upper().split(".")[0] if "." not in symbol else symbol
        if symbol in fixed_map:
            desired_lot = float(fixed_map[symbol])
        elif sym_key in fixed_map:
            desired_lot = float(fixed_map[sym_key])
        else:
            try:
                other_mult = float(getattr(Config, "OTHER_LOT_MULTIPLIER", 1.0))
            except Exception:
                other_mult = 1.0
            desired_lot = desired_lot * other_mult

        # 捨入到 volume_step，clamp
        step = symbol_info.volume_step
        lot  = round(desired_lot / step) * step
        lot  = max(symbol_info.volume_min, min(lot, symbol_info.volume_max, max_lot))

        logger.debug(
            f"[RiskEngine.atr] {symbol}: equity={equity:.0f} atr={atr_price:.4f} "
            f"val_per_unit={value_per_unit:.4f} desired={desired_lot:.4f} lot={lot:.2f}"
        )
        return lot

    def value_per_price_unit(self, symbol: str) -> float:
        """Return account-currency PnL per 1.0 price move for one lot."""
        symbol_info = self._get_symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[RiskEngine.vol] No symbol_info for {symbol}")
            return 0.0

        tick_val = float(symbol_info.trade_tick_value)
        tick_size = float(symbol_info.trade_tick_size)
        if tick_val <= 0 or tick_size <= 0:
            logger.warning(
                f"[RiskEngine.vol] Invalid tick data for {symbol}: "
                f"tick_value={tick_val}, tick_size={tick_size}"
            )
            return 0.0
        return tick_val / tick_size

    def calculate_lot_for_volatility_target(
        self,
        symbol: str,
        atr_price: float,
        target_usd: float,
        exposure: float = 1.0,
        max_lot: float | None = None,
        sharpe_weight: float = 1.0,
    ) -> float:
        """Size lots so one ATR has roughly the requested dollar PnL.

        Unlike the legacy ATR method, this does not clamp undersized requests up
        to volume_min. If the broker minimum is already too large for the risk
        budget, returning 0 is safer than opening an oversized position.
        """
        if atr_price <= 0 or target_usd <= 0:
            logger.warning(
                f"[RiskEngine.vol] Invalid atr={atr_price} target_usd={target_usd} for {symbol}"
            )
            return 0.0

        symbol_info = self._get_symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[RiskEngine.vol] No symbol_info for {symbol}")
            return 0.0

        value_per_unit = self.value_per_price_unit(symbol)
        if value_per_unit <= 0:
            return 0.0

        exposure = max(0.0, min(1.0, float(exposure)))
        sharpe_weight = max(0.0, float(sharpe_weight))
        if exposure <= 0 or sharpe_weight <= 0:
            return 0.0

        desired_lot = (target_usd * exposure * sharpe_weight) / (atr_price * value_per_unit)

        step = float(symbol_info.volume_step)
        volume_min = float(symbol_info.volume_min)
        volume_max = float(symbol_info.volume_max)
        if step <= 0 or volume_min <= 0 or volume_max <= 0:
            logger.warning(f"[RiskEngine.vol] Invalid volume spec for {symbol}")
            return 0.0

        cap = volume_max if max_lot is None else min(volume_max, float(max_lot))
        if desired_lot < volume_min:
            min_risk = atr_price * value_per_unit * volume_min
            logger.warning(
                f"[RiskEngine.vol] {symbol}: desired_lot={desired_lot:.4f} < "
                f"volume_min={volume_min:.4f}; skip to avoid oversizing "
                f"(min_1atr_usd={min_risk:.2f}, target_usd={target_usd:.2f})"
            )
            return 0.0

        # Round down to avoid exceeding the risk budget by rounding up.
        lot = math.floor(desired_lot / step) * step
        lot = max(volume_min, min(lot, cap))
        lot = round(lot, 8)

        logger.info(
            f"[RiskEngine.vol] {symbol}: atr={atr_price:.4f} value_per_unit={value_per_unit:.2f} "
            f"target_usd={target_usd:.2f} exposure={exposure:.2f} weight={sharpe_weight:.2f} "
            f"desired={desired_lot:.4f} lot={lot:.4f}"
        )
        return lot

    # ------------------------------------------------------------------
    # 內部輔助
    # ------------------------------------------------------------------

    def _get_symbol_info(self, symbol: str):
        """封裝 mt5.symbol_info() 調用，便於測試時 mock。"""
        if not _MT5_AVAILABLE or mt5 is None:
            return None
        return mt5.symbol_info(symbol)

    def _get_account_info(self):
        """封裝 mt5.account_info() 調用，便於測試時 mock。"""
        if not _MT5_AVAILABLE or mt5 is None:
            return None
        return mt5.account_info()

    def _has_sufficient_margin(
        self,
        symbol: str,
        lot: float,
        pip_value: float,
        stop_pips: float,
    ) -> bool:
        """檢查帳戶可用保證金是否足夠開倉。

        估算所需保證金 ≈ lot * pip_value * stop_pips。
        若 free_margin < estimated_margin，記錄 WARNING 並返回 False。

        Args:
            symbol:     品種名（用於日誌）
            lot:        已計算手數
            pip_value:  每 pip 價值
            stop_pips:  止損 pips 數

        Returns:
            True 表示保證金充足，False 表示不足。
        """
        acct = self._get_account_info()
        if acct is None:
            # 無法獲取帳戶資訊時保守通過（避免阻塞無 MT5 環境）
            return True

        estimated_margin: float = lot * pip_value * stop_pips
        if acct.margin_free < estimated_margin:
            logger.warning(
                f"[RiskEngine] Insufficient free margin for {symbol}: "
                f"free={acct.margin_free:.2f}, required≈{estimated_margin:.2f}"
            )
            return False

        return True
