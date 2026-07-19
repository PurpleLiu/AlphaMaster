"""下載 BTCUSDT / ETHUSDT 的 H1 全量 K 線（重用 download_okx_klines 的函式）。

用法:
    .venv\Scripts\python download_btc_eth_h1.py --out "..\data\K線資料"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

from download_okx_klines import download_history, fmt_range, save_parquet

TARGETS = [
    ("BTC-USDT-SWAP", "BTCUSDT"),
    ("ETH-USDT-SWAP", "ETHUSDT"),
]
BARS = [("1H", "H1")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for inst_id, symbol in TARGETS:
        for bar, tag in BARS:
            out_path = out_dir / f"{symbol}_{tag}.parquet"
            logger.info(f"下載 {inst_id} {bar} -> {out_path}")
            t0 = time.time()
            df = download_history(inst_id, bar)
            if df.empty:
                logger.error(f"{out_path.name}: 無資料")
                continue
            save_parquet(df, out_path)
            logger.success(
                f"{out_path.name}: {len(df):,} bars  {fmt_range(df)}  ({time.time() - t0:.1f}s)"
            )


if __name__ == "__main__":
    main()
