"""get_spreads.py — 從 MT5 即時獲取各品種點差（用於回測成本）"""
import sys; sys.path.insert(0,'.')
import MetaTrader5 as mt5
from config import Config

mt5.initialize()

print("=== 即時點差 ===")
spreads = {}
for sym in Config.SYMBOLS:
    tick = mt5.symbol_info_tick(sym)
    info = mt5.symbol_info(sym)
    if tick and info:
        spread_pts = tick.ask - tick.bid
        # 轉換為對數收益率單位：spread / mid_price
        mid = (tick.ask + tick.bid) / 2
        spread_pct = spread_pts / mid
        point = info.point
        spread_points = round(spread_pts / point)
        spreads[sym] = {
            "bid": tick.bid, "ask": tick.ask,
            "spread_price": spread_pts,
            "spread_points": spread_points,
            "spread_pct": spread_pct,
            "digits": info.digits,
        }
        print(f"  {sym:12s}: bid={tick.bid:.5f} ask={tick.ask:.5f} "
              f"spread={spread_pts:.5f} ({spread_points}pts) "
              f"log_spread={spread_pct:.6f}")
    else:
        print(f"  {sym:12s}: 無法獲取 tick 數據")

mt5.shutdown()

# 各品種點差的對數收益率（單邊）用於回測 cost_rate
print()
print("=== 回測 cost_rate 建議（log spread / 2，單邊）===")
for sym, d in spreads.items():
    cost = d["spread_pct"] / 2
    print(f"  {sym:12s}: cost_rate = {cost:.6f}")
