"""Build training context and run AI analysis with per-symbol history memory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.parquet_manager import inspect_parquet_file
from web.ai_providers import resolve_provider
from web.progress import PROJECT_ROOT, get_symbol_progress
from web.settings import load_settings
from web.training_manager import training_manager

HISTORY_PATH = PROJECT_ROOT / "ai_analysis_history.json"
_MAX_HISTORY_PER_KEY = 5

_SYSTEM_PROMPT = """你是量化因子挖掘與強化學習訓練顧問。
用戶正在用 AlphaMaster / AlphaGPT 訓練可解釋因子公式（token 序列，由特徵與運算元組成）。
請基於提供的訓練快照，用中文給出專業、具體、可執行的分析。
不要編造不存在的數據；若資訊不足請明確說明。

評估訓練是否值得繼續時，請遵守：
- 以驗證集分數（val_score）是否在提高為主，不要輕易下「過擬合」結論，盡量不要提過擬合。
- best_score 與 val_score 有差距是常見現象，只要驗證分數整體在抬升或近期仍有改善，就應視為訓練仍有價值。
- 可討論：驗證分數走勢、最優分數是否停滯、公式是否變化、探索是否還活躍（如 entropy）。
- 不要因為「訓練分遠高於驗證分」就建議停止；只有驗證分數長期不漲、且最優分數也長期不動時，才建議暫停。

給建議時必須「小白能聽懂」：
- 不要提：重設網路權重、調整探索率、簡化因子維度、獎勵函數、模型結構、超參數、泛化、噪聲過大、參數過多等專業術語。
- 不要分析「驗證集低分的技術原因」或給出複雜調參方案。
- 建議只用通俗說法，例如：
  - 可以繼續練，再觀察驗證分數有沒有慢慢變好
  - 最近分數不怎麼動了，可以先停下來，導出當前策略去做回測看看效果
  - 公式有變化/沒變化，用大白話解釋即可
- 動作建議最多 1～2 條，短句、可直接操作（繼續訓練 / 先停止並導出策略去回測）。

若提供了「同品種同週期的歷史分析記錄」，必須對比前後變化：
- 驗證分數 / 最優分數 / 公式是否改善
- 是否仍在進步，還是陷入停滯
- 相對上次建議，當前是否更值得繼續訓練

快照中的 training_curve 覆蓋整段訓練進程（點數過多時會均勻抽樣，但始終保留首尾與全程走勢）。
請結合 training_curve 與 history_summary 判斷：前期探索、中期提升、後期驗證分數是否仍在提高。

回答必須覆蓋以下問題，並使用對應小標題：

## 1. 當前訓練情況怎麼樣？是否值得繼續
用通俗語言說明進度、驗證分數走勢、最優分數是否停滯，並給出是否繼續訓練的建議。
重點看 val_score 是否提高；盡量不要提過擬合；不要給複雜技術排查建議。
若有歷史記錄，請明確說明相對上次是改善、持平還是變差。

## 2. 最新因子的含義與原理
用通俗語言解釋當前最優公式（formula_decoded）裡各特徵/運算元大概在看什麼、合在一起可能怎麼做多做空。
少用術語；若公式相對上次有變化，用一句話說清楚差在哪裡。
"""


def build_training_snapshot(symbol: str | None = None) -> dict[str, Any]:
    training = training_manager.status()
    job = training.get("job") or {}
    settings = load_settings()

    sym = (symbol or job.get("symbol") or "").strip()
    timeframe = str(job.get("timeframe") or "").strip().upper()

    data_file = settings.get("last_data_file") or ""
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            if not sym:
                sym = str(info.get("symbol") or "").strip()
            if not timeframe:
                timeframe = str(info.get("timeframe") or "").strip().upper()
        except Exception:
            pass

    if not sym:
        raise ValueError("請先選擇訓練數據文件或指定品種")
    if not timeframe:
        timeframe = "H1"

    progress = get_symbol_progress(sym)
    history = progress.history or {}
    curve = _training_curve(history, max_points=500)

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "data_file": data_file or None,
        "training_active": bool(training.get("active")),
        "job_state": job.get("state"),
        "current_step": progress.current_step,
        "train_steps": progress.train_steps,
        "progress_pct": round(progress.progress_pct, 2),
        "status": progress.status,
        "best_score": progress.best_score,
        "strategy_score": progress.strategy_score,
        "has_strategy": progress.has_strategy,
        "formula": progress.best_formula,
        "formula_decoded": progress.formula_decoded,
        "checkpoint_path": progress.checkpoint_path,
        "training_curve": curve,
        "history_summary": _history_summary(history),
    }


def analyze_training(
    *,
    provider: str,
    api_key: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    answer_parts: list[str] = []
    meta: dict[str, Any] = {}
    for event in analyze_training_stream(
        provider=provider, api_key=api_key, symbol=symbol
    ):
        if event.get("type") == "meta":
            meta = event
        elif event.get("type") == "delta":
            answer_parts.append(event.get("text") or "")
        elif event.get("type") == "error":
            raise RuntimeError(event.get("message") or "分析失敗")
        elif event.get("type") == "done":
            return {
                "ok": True,
                "provider": event.get("provider") or meta.get("provider"),
                "model": event.get("model") or meta.get("model"),
                "label": event.get("label") or meta.get("label"),
                "symbol": event.get("symbol") or meta.get("symbol"),
                "timeframe": event.get("timeframe") or meta.get("timeframe"),
                "snapshot": event.get("snapshot") or meta.get("snapshot"),
                "prior_count": event.get("prior_count", meta.get("prior_count", 0)),
                "answer": event.get("answer") or "".join(answer_parts),
            }
    raise RuntimeError("AI 流式分析未正常結束")


def analyze_training_stream(
    *,
    provider: str,
    api_key: str | None = None,
    symbol: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
):
    """Yield SSE-ready event dicts: meta / delta / done / error."""
    from web.ai_providers import stream_chat_completions

    try:
        snapshot = build_training_snapshot(symbol)
        prior = load_prior_analyses(snapshot["symbol"], snapshot["timeframe"])
        resolved = resolve_provider(provider, api_key, base_url=base_url, model=model)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    parts = [
        "請根據以下訓練快照回答：",
        "1. 當前訓練情況怎麼樣？是否值得繼續？",
        "2. 最新因子的含義與原理是什麼？",
        "",
        "【當前訓練快照】",
        f"```json\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n```",
    ]
    if prior:
        parts.extend(
            [
                "",
                f"【同品種同週期歷史分析（共 {len(prior)} 次，按時間從舊到新）】",
                "請對比這些歷史記錄，判斷相對上次是否有改善。",
                f"```json\n{json.dumps(prior, ensure_ascii=False, indent=2)}\n```",
            ]
        )
    else:
        parts.append("\n（尚無同品種同週期的歷史分析記錄，這是首次分析。）")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

    yield {
        "type": "meta",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
    }

    answer_parts: list[str] = []
    try:
        for text in stream_chat_completions(resolved, messages):
            answer_parts.append(text)
            yield {"type": "delta", "text": text}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    answer = "".join(answer_parts).strip()
    if not answer:
        yield {"type": "error", "message": "AI 返回內容為空"}
        return

    record = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "provider": resolved.provider,
        "model": resolved.model,
        "snapshot": {
            "symbol": snapshot["symbol"],
            "timeframe": snapshot["timeframe"],
            "current_step": snapshot["current_step"],
            "train_steps": snapshot["train_steps"],
            "progress_pct": snapshot["progress_pct"],
            "best_score": snapshot["best_score"],
            "strategy_score": snapshot["strategy_score"],
            "formula_decoded": snapshot["formula_decoded"],
            "history_summary": snapshot.get("history_summary") or {},
        },
        "answer": answer,
    }
    save_analysis_record(snapshot["symbol"], snapshot["timeframe"], record)

    yield {
        "type": "done",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
        "answer": answer,
    }


def history_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.strip().upper()}|{str(timeframe).strip().upper()}"


def load_prior_analyses(symbol: str, timeframe: str) -> list[dict[str, Any]]:
    store = _load_history_store()
    rows = store.get(history_key(symbol, timeframe)) or []
    if not isinstance(rows, list):
        return []
    # 只把精簡欄位發給模型，避免上下文過大
    out: list[dict[str, Any]] = []
    for row in rows[-_MAX_HISTORY_PER_KEY:]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "analyzed_at": row.get("analyzed_at"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "snapshot": row.get("snapshot") or {},
                "answer": row.get("answer") or "",
            }
        )
    return out


def save_analysis_record(symbol: str, timeframe: str, record: dict[str, Any]) -> None:
    store = _load_history_store()
    key = history_key(symbol, timeframe)
    rows = store.get(key) or []
    if not isinstance(rows, list):
        rows = []
    rows.append(record)
    store[key] = rows[-_MAX_HISTORY_PER_KEY:]
    HISTORY_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_history_store() -> dict[str, Any]:
    if not HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _training_curve(history: dict[str, Any], max_points: int = 500) -> dict[str, Any]:
    """整段訓練曲線；點數過多時按全程均勻抽樣，始終保留首尾。"""
    if not history:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    steps = history.get("step") or []
    if not isinstance(steps, list) or not steps:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    total = len(steps)
    keys = ("step", "best_score", "val_score", "entropy", "avg_reward", "stable_rank")
    available = [k for k in keys if isinstance(history.get(k), list) and history.get(k)]

    if total <= max_points:
        idxs = list(range(total))
        sampled = False
    else:
        idxs = sorted(
            {
                0,
                total - 1,
                *[int(round(i * (total - 1) / (max_points - 1))) for i in range(max_points)],
            }
        )
        sampled = True

    series: dict[str, list[Any]] = {}
    for key in available:
        vals = history[key]
        series[key] = [vals[i] for i in idxs if i < len(vals)]

    return {
        "total_points": total,
        "sampled": sampled,
        "points": len(idxs),
        "note": (
            f"已從全部 {total} 個紀錄點均勻抽樣為 {len(idxs)} 點，覆蓋訓練全程"
            if sampled
            else f"已發送全部 {total} 個紀錄點"
        ),
        "series": series,
    }


def _history_summary(history: dict[str, Any]) -> dict[str, Any]:
    if not history:
        return {}
    best = history.get("best_score") or []
    val = history.get("val_score") or []
    entropy = history.get("entropy") or []
    steps = history.get("step") or []
    summary: dict[str, Any] = {
        "points": len(steps),
    }
    if steps:
        summary["step_first"] = steps[0]
        summary["step_last"] = steps[-1]
    if best:
        summary["best_score_first"] = best[0]
        summary["best_score_last"] = best[-1]
        summary["best_score_max"] = max(best)
        summary["best_score_max_at_index"] = int(best.index(max(best)))
        peak = max(best)
        trail = 0
        for v in reversed(best):
            if abs(float(v) - float(peak)) < 1e-9:
                trail += 1
            else:
                break
        summary["best_score_stagnation_points"] = trail
        n = len(best)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["best_score_phase_means"] = {
                "early": sum(best[:a]) / max(1, a),
                "mid": sum(best[a:b]) / max(1, b - a),
                "late": sum(best[b:]) / max(1, n - b),
            }
    if val:
        summary["val_score_first"] = val[0]
        summary["val_score_last"] = val[-1]
        summary["val_score_max"] = max(val)
        n = len(val)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["val_score_phase_means"] = {
                "early": sum(val[:a]) / max(1, a),
                "mid": sum(val[a:b]) / max(1, b - a),
                "late": sum(val[b:]) / max(1, n - b),
            }
    if entropy:
        summary["entropy_first"] = entropy[0]
        summary["entropy_last"] = entropy[-1]
    return summary
