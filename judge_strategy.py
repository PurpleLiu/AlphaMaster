"""以命令列執行 AlphaMaster 策略審判。"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping

from judgment.reporting import write_reports
from judgment.runner import run_judgment


class _InputParser(argparse.ArgumentParser):
    """將 argparse 輸入錯誤轉為 CLI 規格要求的 code 1。"""

    def error(self, message: str) -> None:
        raise ValueError(f"輸入錯誤：{message}")


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必須是數字") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("必須是非負且有限的數字")
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必須是正整數") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必須是正整數")
    return parsed


def _non_negative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必須是非負整數") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("必須是非負整數")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """建立台灣正體中文的策略審判命令列介面。"""
    parser = _InputParser(description="執行 AlphaMaster 策略審判")
    parser.add_argument("--strategy", required=True, help="策略 JSON 檔案路徑")
    parser.add_argument("--data", required=True, help="市場 Parquet K 線資料路徑")
    parser.add_argument(
        "--base-cost",
        type=_non_negative_float,
        default=0.0006,
        help="單邊交易成本，例如 0.0006 代表 0.06%%",
    )
    parser.add_argument(
        "--equal-blocks", type=_positive_integer, default=8, help="時間等分區塊數"
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=_positive_integer,
        default=5,
        help="偽樣本外切分數",
    )
    parser.add_argument(
        "--embargo-bars",
        type=_non_negative_integer,
        default=24,
        help="偽樣本外資料隔離 K 線數",
    )
    parser.add_argument("--output-dir", default="reports/judgment", help="報告輸出資料夾")
    return parser


def _verdict_status(result: object) -> str | None:
    if not isinstance(result, Mapping):
        return None
    verdict = result.get("verdict")
    if not isinstance(verdict, Mapping):
        return None
    status = verdict.get("status")
    return status if isinstance(status, str) else None


def main(argv: list[str] | None = None) -> int:
    """執行策略審判並回傳可供自動化使用的 exit code。"""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    try:
        result = run_judgment(
            args.strategy,
            args.data,
            base_cost=args.base_cost,
            equal_blocks=args.equal_blocks,
            walk_forward_splits=args.walk_forward_splits,
            embargo_bars=args.embargo_bars,
        )
        status = _verdict_status(result)
        exit_code = {"PASS": 0, "REVIEW": 2, "FAIL": 3}.get(status)
        if exit_code is None:
            raise ValueError(f"未知的審判結果：{status!r}")
        json_path, markdown_path = write_reports(result, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"執行失敗：{error}", file=sys.stderr)
        return 1

    print(f"審判結果：{status}")
    print(f"JSON 報告：{json_path}")
    print(f"Markdown 報告：{markdown_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
