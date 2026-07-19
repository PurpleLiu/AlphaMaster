"""
download_okx_klines.py — 從 OKX 下載主流 USDT 永續合約 K 線

輸出格式與 D:\\OKX_K線數據 一致：
    {品種}_{週期}.parquet
    列: time, open, high, low, close, tick_volume
    time 為 Unix 秒（int64）

週期映射:
    5m -> M5, 15m -> M15, 1H -> H1, 1D -> D1

主流幣名單（20 隻，按市值/流動性常見排序，不含穩定幣）:
    BTC ETH XRP BNB SOL DOGE ADA LINK BCH XLM
    LTC HBAR AVAX UNI DOT POL SHIB NEAR ATOM TRX

用法:
    python download_okx_klines.py
    python download_okx_klines.py --out D:\\OKX_K線數據
    python download_okx_klines.py --resume

每個合約/週期會拉取 OKX 能提供的全部歷史 K 線（不設條數上限）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from config import Config

OKX_BASE = "https://www.okx.com"
DEFAULT_OUT = Path(r"D:\OKX_K線數據")
REQUEST_SLEEP = 0.21
LOG_EVERY_PAGES = 50

# 固定主流 20 幣（參考 CoinMarketCap 市值前列，剔除 USDT/USDC 等穩定幣）
MAINSTREAM_BASES: tuple[str, ...] = (
    "BTC", "ETH", "XRP", "BNB", "SOL",
    "DOGE", "ADA", "LINK", "BCH", "XLM",
    "LTC", "HBAR", "AVAX", "UNI", "DOT",
    "POL", "SHIB", "NEAR", "ATOM", "TRX",
)

TIMEFRAMES: dict[str, str] = {
    "5m": "M5",
    "15m": "M15",
    "1H": "H1",
    "1D": "D1",
}

_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


def okx_get(path: str, params: dict[str, str], retries: int = 5) -> list:
    query = urllib.parse.urlencode(params)
    url = f"{OKX_BASE}{path}?{query}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaMaster/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("code") != "0":
                raise RuntimeError(f"OKX API {body.get('code')}: {body.get('msg')}")
            return body.get("data") or []
        except Exception as exc:
            last_err = exc
            wait = min(60, 2 ** attempt)
            logger.warning(f"請求失敗 ({attempt + 1}/{retries}): {exc}，{wait}s 後重試")
            time.sleep(wait)
    raise RuntimeError(f"OKX 請求失敗: {last_err}")


def inst_to_symbol(inst_id: str) -> str:
    """BTC-USDT-SWAP -> BTCUSDT"""
    if not inst_id.endswith("-USDT-SWAP"):
        raise ValueError(f"非 USDT 永續: {inst_id}")
    base = inst_id[: -len("-USDT-SWAP")]
    return f"{base}USDT"


def fetch_available_usdt_swaps() -> set[str]:
    instruments = okx_get("/api/v5/public/instruments", {"instType": "SWAP"})
    return {
        str(x.get("instId", ""))
        for x in instruments
        if str(x.get("instId", "")).endswith("-USDT-SWAP")
    }


def fetch_mainstream_swap_inst_ids() -> list[str]:
    available = fetch_available_usdt_swaps()
    inst_ids: list[str] = []
    missing: list[str] = []

    for base in MAINSTREAM_BASES:
        inst_id = f"{base}-USDT-SWAP"
        if inst_id in available:
            inst_ids.append(inst_id)
        else:
            missing.append(base)

    if missing:
        logger.warning(f"以下主流幣在 OKX 無 USDT 永續，已跳過: {', '.join(missing)}")
    if not inst_ids:
        raise RuntimeError("未找到任何可用的主流 USDT 永續合約")

    logger.info(f"已選取固定主流幣 {len(inst_ids)} 只 USDT 永續")
    for i, inst in enumerate(inst_ids, 1):
        logger.info(f"  #{i:02d} {inst}")
    return inst_ids


def download_history(inst_id: str, bar: str, max_bars: int | None = None) -> pd.DataFrame:
    rows: list[list[str]] = []
    after: str | None = None
    stagnant = 0
    page = 0

    while max_bars is None or len(rows) < max_bars:
        params: dict[str, str] = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = after
        batch = okx_get("/api/v5/market/history-candles", params)
        if not batch:
            break

        prev_len = len(rows)
        rows.extend(batch)
        page += 1
        if page % LOG_EVERY_PAGES == 0:
            logger.info(f"    {inst_id} {bar}: 已拉取 {len(rows):,} 根…")

        if len(rows) == prev_len:
            break

        oldest_ts = batch[-1][0]
        if after is not None and oldest_ts == after:
            stagnant += 1
            if stagnant >= 2:
                break
        else:
            stagnant = 0
        after = oldest_ts

        if len(batch) < 100:
            break
        time.sleep(REQUEST_SLEEP)

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    records = []
    for item in rows:
        records.append(
            {
                "time": int(int(item[0]) // 1000),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "tick_volume": int(float(item[5])),
            }
        )

    df = pd.DataFrame(records)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if max_bars is not None and len(df) > max_bars:
        df = df.iloc[-max_bars:].reset_index(drop=True)

    return df.astype(
        {
            "time": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "tick_volume": "int64",
        }
    )


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"done": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"done": []}


def manifest_key(symbol: str, tf_tag: str) -> str:
    return f"{symbol}_{tf_tag}"


def fmt_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "無數據"
    t0 = datetime.fromtimestamp(int(df["time"].iloc[0]), tz=timezone.utc).strftime("%Y-%m-%d")
    t1 = datetime.fromtimestamp(int(df["time"].iloc[-1]), tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{t0} ~ {t1}"


def run(
    out_dir: Path,
    resume: bool,
    max_bars: int | None = None,
) -> None:
    manifest_path = out_dir / "okx_download_manifest.json"
    manifest = load_manifest(manifest_path)
    done: set[str] = set(manifest.get("done") or [])

    inst_ids = fetch_mainstream_swap_inst_ids()
    total_tasks = len(inst_ids) * len(TIMEFRAMES)
    finished = 0
    ok = 0
    skipped = 0
    failed = 0

    logger.info(f"輸出目錄: {out_dir}")
    logger.info(
        f"歷史範圍: {'全量（OKX 能提供的全部 K 線）' if max_bars is None else f'最多 {max_bars:,} 根'}"
    )
    logger.info(f"任務總數: {total_tasks}（{len(inst_ids)} 品種 × {len(TIMEFRAMES)} 週期）")

    for inst_id in inst_ids:
        symbol = inst_to_symbol(inst_id)
        for bar, tf_tag in TIMEFRAMES.items():
            finished += 1
            key = manifest_key(symbol, tf_tag)
            out_path = out_dir / f"{symbol}_{tf_tag}.parquet"

            if resume and key in done and out_path.exists():
                skipped += 1
                logger.info(f"[{finished}/{total_tasks}] 跳過已完成 {out_path.name}")
                continue

            logger.info(f"[{finished}/{total_tasks}] 下載 {inst_id} {bar} -> {out_path.name}")
            t0 = time.time()
            try:
                df = download_history(inst_id, bar, max_bars)
                if df.empty:
                    logger.warning(f"  {out_path.name}: 無數據")
                    failed += 1
                    continue
                save_parquet(df, out_path)
                done.add(key)
                manifest["done"] = sorted(done)
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                ok += 1
                logger.success(
                    f"  {out_path.name}: {len(df):,} bars  {fmt_range(df)}  ({time.time() - t0:.1f}s)"
                )
                if len(df) < Config.MIN_BARS:
                    logger.warning(
                        f"  {out_path.name}: 僅 {len(df)} 根，低於訓練最低要求 {Config.MIN_BARS}"
                    )
            except Exception as exc:
                failed += 1
                logger.error(f"  {out_path.name}: 失敗 — {exc}")

    logger.info(
        f"完成。成功 {ok}，跳過 {skipped}，失敗 {failed}，總計 {total_tasks}。"
        f"清單: {manifest_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下載 OKX 主流合約 K 線到 Parquet")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="輸出目錄")
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="可選：限制每個文件最多保留 K 線根數（默認不限制，拉全歷史）",
    )
    parser.add_argument("--resume", action="store_true", help="跳過 manifest 中已完成的文件")
    args = parser.parse_args()

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "okx_download.log", rotation="20 MB", encoding="utf-8")

    print(f"{'=' * 62}")
    print("  OKX 主流幣 K 線下載")
    print(f"  固定 {len(MAINSTREAM_BASES)} 只: {' '.join(MAINSTREAM_BASES)}")
    print(f"  週期 {', '.join(TIMEFRAMES.values())}")
    print(f"  歷史: {'全量' if args.max_bars is None else f'最多 {args.max_bars:,} 根'}")
    print(f"  保存至: {args.out}")
    print(f"{'=' * 62}\n")

    run(
        out_dir=Path(args.out),
        resume=args.resume,
        max_bars=args.max_bars,
    )


if __name__ == "__main__":
    main()
