"""即時信號計算：因子公式 + 即時 K 線 → 方向 + 強度。

與回測走完全相同的計算鏈（compute_features → StackVM → tanh → 閾值），
保證即時信號與回測/訓練目標一致。信號取最後一根已收盤 bar。
"""
from __future__ import annotations

import math
from typing import Any

import torch

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM

# 與回測/實盤共用的無信號閾值（Config.MIN_TRADE_EXPOSURE）
try:
    from config import Config
    _MIN_EXPOSURE = float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05))
except Exception:  # noqa: BLE001
    _MIN_EXPOSURE = 0.05

# 特徵滾動窗口需要足夠歷史才能穩定（_NORM_WINDOW=200 等）
MIN_BARS = 200

_VM = StackVM()

DIR_LONG = "LONG"
DIR_SHORT = "SHORT"
DIR_FLAT = "FLAT"


def min_exposure() -> float:
    return _MIN_EXPOSURE


def evaluate_signal(formula: list[int], raw_dict: dict[str, Any]) -> dict[str, Any]:
    """在即時 K 線上計算因子信號。

    Args:
        formula:  策略因子的 token 序列。
        raw_dict: {open,high,low,close,volume} torch 張量 [1, T]，升序。

    Returns:
        dict：state / direction / strength / factor_value / position / bars_used / message
    """
    close = raw_dict.get("close")
    if close is None or close.ndim != 2:
        return {"state": "error", "message": "行情數據格式無效"}

    n_bars = int(close.shape[1])
    if n_bars < MIN_BARS:
        return {
            "state": "insufficient",
            "bars_used": n_bars,
            "message": f"歷史 bar 不足（{n_bars}/{MIN_BARS}），無法穩定計算特徵",
        }

    try:
        feats = MT5FeatureEngineer.compute_features(raw_dict)  # [1, F, T]
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "bars_used": n_bars, "message": f"特徵計算失敗: {exc}"}

    try:
        factor = _VM.execute([int(t) for t in formula], feats)  # [1, T] or None
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "bars_used": n_bars, "message": f"公式執行失敗: {exc}"}

    if factor is None or factor.ndim != 2 or factor.shape[1] == 0:
        return {"state": "error", "bars_used": n_bars, "message": "公式無有效輸出"}

    factor_last = float(factor[0, -1])
    if not math.isfinite(factor_last):
        return {"state": "error", "bars_used": n_bars, "message": "因子值非有限"}

    position = math.tanh(factor_last)          # 連續倉位 [-1, 1]
    strength = abs(position)                    # 信號強度 [0, 1]
    thr = _MIN_EXPOSURE

    if position >= thr:
        direction = DIR_LONG
    elif position <= -thr:
        direction = DIR_SHORT
    else:
        direction = DIR_FLAT

    return {
        "state": "ok",
        "direction": direction,
        "strength": round(strength, 4),
        "position": round(position, 4),
        "factor_value": round(factor_last, 6),
        "threshold": thr,
        "bars_used": n_bars,
        "message": "",
    }
