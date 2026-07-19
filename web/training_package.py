"""Export / import training checkpoints as portable zip packages."""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from model_core.vocab import FORMULA_VOCAB
from web.progress import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    _decode_formula,
    _load_strategy,
    checkpoint_glob,
    invalidate_checkpoint_cache,
)

_CKPT_NAME_RE = re.compile(r"^ckpt_(.+)_step_(\d+)\.pt$", re.IGNORECASE)


def _history_path(symbol: str) -> Path:
    return PROJECT_ROOT / f"training_history_{symbol}.json"


def _symbol_from_ckpt_name(name: str) -> str | None:
    m = _CKPT_NAME_RE.match(Path(name).name)
    return m.group(1) if m else None


def _validate_checkpoint_file(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    artifact_version = ckpt.get("vocab_version")
    if artifact_version is None:
        raise ValueError(
            f"檢查點 {path.name} 過舊（無 vocab_version），"
            f"當前詞表 {FORMULA_VOCAB.version!r}，請重新訓練"
        )
    FORMULA_VOCAB.verify(artifact_version)
    step = int(ckpt.get("step", 0))
    symbol = ckpt.get("symbol")
    return {"step": step, "symbol": symbol, "best_score": ckpt.get("best_score")}


def build_training_export_zip(symbol: str) -> tuple[bytes, str]:
    ckpts = checkpoint_glob(symbol)
    if not ckpts:
        raise FileNotFoundError(f"未找到 {symbol} 的訓練檢查點，請先訓練並保存 checkpoint")

    latest = ckpts[-1]
    _validate_checkpoint_file(latest)

    strategy = _load_strategy(symbol)
    history_path = _history_path(symbol)

    step = int(re.search(r"_step_(\d+)\.pt$", latest.name).group(1))
    safe = symbol.replace(".", "_")
    zip_name = f"training_{safe}_step{step:04d}.zip"

    manifest = {
        "format": "alphamaster_training_v1",
        "symbol": symbol,
        "step": step,
        "checkpoint": f"checkpoints/{latest.name}",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": [f"checkpoints/{latest.name}"],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(latest, f"checkpoints/{latest.name}")

        if strategy:
            strat_name = f"strategies/best_{symbol}.json"
            payload = dict(strategy)
            if payload.get("formula") and not payload.get("formula_decoded"):
                payload["formula_decoded"] = _decode_formula(payload["formula"])
            zf.writestr(strat_name, json.dumps(payload, ensure_ascii=False, indent=2))
            manifest["files"].append(strat_name)

        if history_path.exists():
            hist_name = f"training_history_{symbol}.json"
            zf.write(history_path, hist_name)
            manifest["files"].append(hist_name)

        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return buf.getvalue(), zip_name


def _remove_symbol_checkpoints(symbol: str) -> int:
    removed = 0
    for path in checkpoint_glob(symbol):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def import_training_package(
    content: bytes,
    filename: str,
    expected_symbol: str | None = None,
) -> dict[str, Any]:
    name = Path(filename).name.lower()
    installed: list[str] = []
    symbol: str | None = None
    step: int | None = None

    if name.endswith(".zip"):
        result = _import_zip(content, expected_symbol)
        symbol = result["symbol"]
        step = result["step"]
        installed = result["installed"]
    elif name.endswith(".pt"):
        result = _import_pt(content, filename, expected_symbol)
        symbol = result["symbol"]
        step = result["step"]
        installed = result["installed"]
    else:
        raise ValueError("僅支持 .zip 訓練包或 .pt 檢查點文件")

    invalidate_checkpoint_cache()
    return {
        "ok": True,
        "symbol": symbol,
        "step": step,
        "installed": installed,
        "message": f"已導入 {symbol} 的訓練文件（step {step}），下次訓練將從斷點續訓",
    }


def _import_zip(content: bytes, expected_symbol: str | None) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if "manifest.json" not in names:
            raise ValueError("訓練包缺少 manifest.json")

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "alphamaster_training_v1":
            raise ValueError("不支持的訓練包格式")

        symbol = manifest.get("symbol") or ""
        ckpt_rel = manifest.get("checkpoint", "")
        if not symbol or not ckpt_rel:
            raise ValueError("訓練包 manifest 不完整")

        if expected_symbol and symbol != expected_symbol:
            raise ValueError(
                f"訓練包品種為 {symbol}，與當前選擇的 {expected_symbol} 不一致"
            )

        _remove_symbol_checkpoints(symbol)
        installed: list[str] = []
        imported_history = False

        for member in names:
            if member == "manifest.json":
                continue
            dest = PROJECT_ROOT / member.replace("/", "\\") if "\\" in str(PROJECT_ROOT) else PROJECT_ROOT / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            installed.append(str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/"))

            if member.endswith(".pt"):
                _validate_checkpoint_file(dest)
            if member == f"training_history_{symbol}.json":
                imported_history = True

        # 若 zip 不包含訓練曲線，項目中可能殘留更“新”的 history，導致 UI 顯示步數不變。
        # 這類 zip 通常只有 checkpoint + strategy（例如只想遷移斷點），因此應清掉舊曲線以與 checkpoint 對齊。
        if not imported_history:
            try:
                _history_path(symbol).unlink(missing_ok=True)
            except OSError:
                pass

        step = int(manifest.get("step") or 0)
        return {"symbol": symbol, "step": step, "installed": installed}


def _import_pt(content: bytes, filename: str, expected_symbol: str | None) -> dict[str, Any]:
    ckpt_name = Path(filename).name
    symbol = _symbol_from_ckpt_name(ckpt_name)
    if not symbol:
        raise ValueError(f"無法從檔案名識別品種: {ckpt_name}（須為 ckpt_品種_step_XXXX.pt）")

    if expected_symbol and symbol != expected_symbol:
        raise ValueError(
            f"檢查點品種為 {symbol}，與當前選擇的 {expected_symbol} 不一致"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / ckpt_name

    # 只導入 .pt 時，項目裡可能還殘留更“新”的 training_history_{symbol}.json。
    # Web 進度會優先使用該曲線文件的最後一步，導致顯示步數大於 checkpoint。
    # 因此這裡主動清掉舊曲線，避免「導入 60 步卻顯示 84」。
    try:
        _history_path(symbol).unlink(missing_ok=True)
    except OSError:
        pass

    _remove_symbol_checkpoints(symbol)
    dest.write_bytes(content)
    meta = _validate_checkpoint_file(dest)

    return {
        "symbol": symbol,
        "step": meta["step"],
        "installed": [str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/")],
    }
