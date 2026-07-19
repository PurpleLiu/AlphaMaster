"""
strategy_manager/signal.py — 回測與實盤共享的信號計算模組

提供：
  compute_target_positions(factors, prev_positions)  →  連續倉位 [-1, +1] 張量
  reconcile_action(current, target)                  →  動作字串

信號邏輯（收益優先模式，2026-07-04 重構）：
  舊模式（Neutral Band）：tanh → sign → {-1, 0, +1} 三檔，天花板鎖死在 1 倍倉。
  新模式（連續倉位）：factor 直接經 tanh 壓縮到 (-1, +1) 作為倉位比例。
    - factor 越強 → 倉位比例越大，允許"加碼"
    - 不設 Neutral Band，讓模型自由決定在場時間
    - 回測與實盤共用同一邏輯，消除兩者差異
  訓練時用 tanh(factor) 作為連續倉位，回測也一致，避免訓練/回測目標函數不對齊。
"""
from __future__ import annotations

import torch
from torch import Tensor

# ── 保留實盤用的閾值參數（實盤 Runner 可能還讀取這些常量）──────────────────
ENTRY_THRESHOLD: float = 0.3
EXIT_THRESHOLD:  float = 0.1
MIN_TRADE_EXPOSURE: float = 0.05


def _min_trade_exposure() -> float:
    try:
        from config import Config
        return float(getattr(Config, "MIN_TRADE_EXPOSURE", MIN_TRADE_EXPOSURE))
    except Exception:
        return MIN_TRADE_EXPOSURE


def compute_target_positions(
    factors:        Tensor,
    prev_positions: Tensor | None = None,
) -> Tensor:
    """將因子張量轉換為連續倉位 [-1, +1]（收益優先模式）。

    新邏輯：position = tanh(factor)，連續倉位，強信號→大倉。
    prev_positions 參數保留相容性，連續模式下不影響計算。

    Args:
        factors:        [N, T] 或 [N] 的因子張量。
        prev_positions: 保留參數，連續模式下忽略。
    """
    pos = torch.tanh(factors)
    min_abs = _min_trade_exposure()
    if min_abs > 0:
        pos = torch.where(pos.abs() >= min_abs, pos, torch.zeros_like(pos))
    return pos

def compute_target_positions_stateless(factors: Tensor) -> Tensor:
    """無狀態版本，供訓練回測快速計算（連續倉位模式）。"""
    return compute_target_positions(factors, prev_positions=None)


def target_to_direction(target: float, min_abs: float | None = None) -> int:
    """把連續目標倉位轉成 MT5 可執行方向。"""
    threshold = _min_trade_exposure() if min_abs is None else float(min_abs)
    if target >= threshold:
        return 1
    if target <= -threshold:
        return -1
    return 0


# ── 動作常量 ──────────────────────────────────────────────────────────────────
HOLD             = "HOLD"
OPEN_LONG        = "OPEN_LONG"
OPEN_SHORT       = "OPEN_SHORT"
CLOSE            = "CLOSE"
REVERSE_TO_LONG  = "REVERSE_TO_LONG"
REVERSE_TO_SHORT = "REVERSE_TO_SHORT"


def reconcile_action(current: int, target: int) -> str:
    """根據當前倉位方向和目標方向，返回應執行的動作。

    Args:
        current: 當前倉位方向，+1（多）/ -1（空）/ 0（空倉）。
        target:  目標倉位方向，+1 / -1 / 0。

    Returns:
        動作字串，取值為模組級常量之一。
    """
    if current == target:
        return HOLD
    if current == 0:
        return OPEN_LONG if target == 1 else OPEN_SHORT
    if target == 0:
        return CLOSE
    return REVERSE_TO_LONG if target == 1 else REVERSE_TO_SHORT
