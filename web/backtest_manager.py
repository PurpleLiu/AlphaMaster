"""Subprocess manager for run_backtest.py jobs.

鏡像 training_manager 的設計：用子進程運行 run_backtest.py，
把 stdout 寫入 logs/backtest_*.log，前端通過輪詢讀取尾部日誌與階段進度。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 回測階段：用日誌關鍵字推斷當前進行到哪一步，用於前端進度展示
BACKTEST_PHASES: list[tuple[str, str]] = [
    ("init", "初始化"),
    ("cost", "交易成本"),
    ("strategy", "載入策略"),
    ("data", "載入行情數據"),
    ("compute", "回測計算"),
    ("chart", "生成圖表"),
    ("done", "完成"),
]
_PHASE_KEYS = [p[0] for p in BACKTEST_PHASES]


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class BacktestJob:
    strategy_file: str
    symbol: str
    commission_pct: float = 0.02
    slippage_pct: float = 0.01
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_file": self.strategy_file,
            "symbol": self.symbol,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class BacktestManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: BacktestJob | None = None
        self._log_fp = None
        self._stopped_by_user = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            job_dict = self._job.to_dict() if self._job else None
        phase_key, phase_label, phase_idx = self._current_phase()
        return {
            "active": self._job is not None and self._job.state == JobState.RUNNING,
            "job": job_dict,
            "phase": phase_key,
            "phase_label": phase_label,
            "phase_index": phase_idx,
            "phase_total": len(BACKTEST_PHASES),
        }

    def start(
        self,
        strategy_file: str,
        data_file: str | None = None,
        commission_pct: float = 0.02,
        slippage_pct: float = 0.01,
    ) -> BacktestJob:
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有回測任務在運行")

            from web.strategy_file import inspect_strategy_file

            info = inspect_strategy_file(strategy_file)
            symbol = info.get("symbol") or ""

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"backtest_{ts}.log"

            cmd = [
                sys.executable,
                "-u",
                "run_backtest.py",
                "--strategy-file",
                strategy_file,
                "--commission",
                str(commission_pct),
                "--slippage",
                str(slippage_pct),
            ]
            if not data_file:
                raise RuntimeError(
                    "回測必須使用本地 Parquet（策略未記錄 data_file，且未傳入數據文件）"
                )
            cmd.extend(["--data-file", data_file])

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._job = BacktestJob(
                strategy_file=strategy_file,
                symbol=symbol,
                commission_pct=float(commission_pct),
                slippage_pct=float(slippage_pct),
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            return self._job

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            try:
                self._proc.terminate()
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            return True

    def _current_phase(self) -> tuple[str, str, int]:
        """根據日誌內容推斷當前回測階段。"""
        lines = self.tail_log(200)
        if not lines:
            return ("init", "初始化", 0)
        text = "\n".join(lines)
        # 從後往前匹配最靠後的階段關鍵字
        detected = "init"
        if "交易成本" in text or "手續費=" in text:
            detected = "cost"
        if "載入各品種策略" in text or re.search(r"score=", text) or "模式:" in text:
            detected = "strategy"
        if "正在載入數據" in text:
            detected = "data"
        if re.search(r"品種:\s*\[", text) or "多因子回測報告" in text:
            detected = "compute"
        if "生成 K 線圖" in text or "張縮放圖" in text:
            detected = "chart"
        if "完成。" in text or "JSON 報告已保存" in text:
            detected = "done"
        idx = _PHASE_KEYS.index(detected) if detected in _PHASE_KEYS else 0
        label = BACKTEST_PHASES[idx][1]
        return (detected, label, idx)

    def tail_log(self, lines: int = 200) -> list[str]:
        with self._lock:
            if not self._job or not self._job.log_path:
                return []
            path = PROJECT_ROOT / self._job.log_path
            if not path.exists():
                return []
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return []
            return [strip_ansi(line) for line in content.splitlines()[-lines:]]

    def _refresh_state(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._job.state == JobState.RUNNING:
            if self._stopped_by_user:
                self._job.state = JobState.STOPPED
            elif code == 0:
                self._job.state = JobState.COMPLETED
            elif code < 0:
                self._job.state = JobState.STOPPED
            else:
                self._job.state = JobState.FAILED
        if self._job.state == JobState.FAILED and self._job.error is None:
            self._job.error = f"回測進程異常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 回測進程已結束，退出碼: {code}\n")
            except OSError:
                pass
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._proc = None


backtest_manager = BacktestManager()
