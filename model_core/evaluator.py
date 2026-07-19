"""
model_core/evaluator.py -- Effectiveness_Evaluator（R4–R7）

實現候選因子/運算元的打分、相關性剪枝、消融、排序報告與 Active_Subset 選擇。

設計核心原則（conservative pruning / 寧可放過，不要錯殺）：
  - 打分退化不等於剔除：unscorable 候選仍保留在報告中，哨兵排序墊底；
  - 相關性閾值偏高（默認 0.9，conservative=True 上調到 0.95）；
  - 剪枝需雙條件（|corr|≥threshold AND 分差>margin），勢均力敵時保留；
  - 消融 drop 只是建議標記，帶負值保護帶（默認 -0.01），不物理刪除；
  - Active_Subset 默認全保留（retention_threshold=0.0, max_retained=None）。
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional

import torch


# ── 異常類型 ─────────────────────────────────────────────────────────────

class ConfigError(Exception):
    """配置參數越界（如 corr_threshold ∉ [0,1]），保留原狀態（R5.5）。"""


class ReportPersistError(Exception):
    """報告持久化失敗，保留舊文件並拋出（R7.6）。"""



# ── 數據模型 ─────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """單個候選的打分結果（R4）。

    importance_score 為聚合分數；unscorable=True 表示序列退化/樣本不足，
    importance_score 此時為 -inf（哨兵，排序用，絕不產 NaN）。
    """
    candidate: str
    category: str
    ic: float          # IC 均值（時間維 Pearson 的品種平均）
    rank_ic: float     # RankIC（Spearman）
    ir: float          # IC / std(IC)，跨時間分塊
    mi: float          # 互資訊（等頻分箱）
    importance_score: float   # 聚合分數（秩歸一加權），或 -inf
    unscorable: bool   # True = 退化/樣本不足，保留但排序墊底


@dataclass
class ReportRow(ScoreResult):
    """打分結果擴展：含剪枝狀態與消融結果（R7.2）。"""
    retention_status: str                    # "retained" | "pruned"
    pruned_in_favor_of: Optional[str]        # 被剪時的保留代表，否則 None
    marginal_contribution: Optional[float]   # None → 序列化為 "not_computed"
    drop_recommendation: bool


@dataclass
class AblationResult:
    """單元素消融結果（R6）。"""
    name: str
    marginal_contribution: Optional[float]
    drop_recommendation: bool
    error: Optional[str]   # None=成功；否則描述失敗配置


@dataclass
class Report:
    """排序報告（R7.5）。"""
    vocab_version: str
    generated_at: str      # ISO-8601
    config: dict
    rows: list[ReportRow]
    active_subset: list[str]


# ── 內部工具函數 ───────────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor):
    """將 Tensor 轉為 numpy（detach + cpu）。"""
    return t.detach().cpu().float().numpy()


def _align_causal(
    candidate: torch.Tensor,
    target: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """防 look-ahead 的時間對齊（R4.7）。

    candidate: [N, T] 或 [1, T]
    target:    [N, T]（前視收益，target[t] 代表 t→t+h 的未來收益）

    配對規則：把 candidate[t] 與 target[t]（代表 t→t+h）配對，
    末端裁掉最後 h 步以避免用到尚未實現的收益。
    返回 (cand_aligned, tgt_aligned)，形狀 [N, T-h]。

    嚴格防止 look-ahead：只使用 ≤t 的 candidate 值配對 target[t]，
    而非 target[t+h]——因為 target[t] 本身就代表 t→t+h 的收益。
    """
    horizon = max(1, int(horizon))
    T = target.shape[-1]
    if T <= horizon:
        # 無法裁剪，返回空對
        empty_c = candidate[..., :0]
        empty_t = target[..., :0]
        return empty_c, empty_t
    # 裁掉末端 h 步：candidate 和 target 均取 [0, T-h)
    cand_aligned = candidate[..., : T - horizon]
    tgt_aligned = target[..., : T - horizon]
    return cand_aligned, tgt_aligned


def _compute_ic_rankic(
    cand: torch.Tensor,
    tgt: torch.Tensor,
) -> tuple[float, float, float]:
    """計算 IC（Pearson）和 RankIC（Spearman），返回 (ic_mean, rank_ic_mean, ir)。

    cand: [N, T_eff]，tgt: [N, T_eff]
    對每個品種計算時間維相關係數，再跨品種取均值。
    IR = mean(IC) / std(IC)（跨品種 IC 的變異作為跨時間分塊的近似）。
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr  # type: ignore

    cand_np = _to_numpy(cand)   # [N, T_eff]
    tgt_np = _to_numpy(tgt)     # [N, T_eff]
    N, T = cand_np.shape

    ic_list = []
    rank_ic_list = []

    for i in range(N):
        c = cand_np[i]
        t_ = tgt_np[i]
        # 僅取有效（有限）重疊樣本
        mask = np.isfinite(c) & np.isfinite(t_)
        if mask.sum() < 2:
            continue
        cv, tv = c[mask], t_[mask]
        # 檢查方差
        if cv.std() < 1e-8 or tv.std() < 1e-8:
            continue
        try:
            ic_val, _ = pearsonr(cv, tv)
            ric_val, _ = spearmanr(cv, tv)
        except Exception:
            continue
        if math.isfinite(ic_val):
            ic_list.append(ic_val)
        if math.isfinite(ric_val):
            rank_ic_list.append(ric_val)

    if len(ic_list) == 0:
        return 0.0, 0.0, 0.0

    ic_mean = float(np.mean(ic_list))
    ric_mean = float(np.mean(rank_ic_list)) if rank_ic_list else 0.0
    # IR = mean / std；std=0 時 IR=0
    ic_std = float(np.std(ic_list))
    ir = ic_mean / (ic_std + 1e-10) if ic_std > 1e-10 else 0.0
    return ic_mean, ric_mean, ir


def _compute_mi(
    cand: torch.Tensor,
    tgt: torch.Tensor,
) -> float:
    """計算離散互資訊（等頻分箱，自適應箱數）。

    箱數 = max(5, min(20, int(sqrt(T_eff))))。
    返回 MI 值（≥0），序列退化或樣本不足時返回 0.0。
    """
    import numpy as np

    cand_np = _to_numpy(cand)   # [N, T_eff]
    tgt_np = _to_numpy(tgt)     # [N, T_eff]
    N, T = cand_np.shape

    # 合併所有品種的樣本（簡化：跨品種池化）
    c_flat = cand_np.flatten()
    t_flat = tgt_np.flatten()

    mask = np.isfinite(c_flat) & np.isfinite(t_flat)
    n_valid = mask.sum()
    if n_valid < 2:
        return 0.0

    cv, tv = c_flat[mask], t_flat[mask]

    if cv.std() < 1e-8:
        return 0.0

    n_bins = max(5, min(20, int(math.sqrt(n_valid))))

    # 等頻分箱
    try:
        c_q = np.percentile(cv, np.linspace(0, 100, n_bins + 1))
        t_q = np.percentile(tv, np.linspace(0, 100, n_bins + 1))
        c_q = np.unique(c_q)
        t_q = np.unique(t_q)
        c_bins = np.searchsorted(c_q[1:-1], cv, side="right")
        t_bins = np.searchsorted(t_q[1:-1], tv, side="right")
    except Exception:
        return 0.0

    # 聯合/邊緣機率
    n_c = len(c_q)
    n_t = len(t_q)
    joint = np.zeros((n_c, n_t), dtype=np.float64)
    for ci, ti in zip(c_bins, t_bins):
        joint[ci, ti] += 1.0
    joint /= joint.sum() + 1e-15

    pc = joint.sum(axis=1)
    pt = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_c):
        for j in range(n_t):
            pij = joint[i, j]
            if pij > 0 and pc[i] > 0 and pt[j] > 0:
                mi += pij * math.log(pij / (pc[i] * pt[j]))

    return max(0.0, float(mi))


def _is_degenerate(cand: torch.Tensor) -> bool:
    """判斷候選序列是否退化（std < 1e-8 或有效樣本 < 2）。"""
    import numpy as np
    c = _to_numpy(cand).flatten()
    mask = np.isfinite(c)
    if mask.sum() < 2:
        return True
    valid = c[mask]
    return float(valid.std()) < 1e-8


def _pearson_corr(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """計算兩個序列的 Pearson 相關（絕對值），重疊有效樣本。

    重疊 < 2 或任一方零方差 → 返回 0.0（視為不相關，保守不剪，R5.8）。
    """
    import numpy as np
    from scipy.stats import pearsonr  # type: ignore

    a_np = _to_numpy(a).flatten()
    b_np = _to_numpy(b).flatten()

    # 對齊長度
    n = min(len(a_np), len(b_np))
    a_np, b_np = a_np[:n], b_np[:n]

    mask = np.isfinite(a_np) & np.isfinite(b_np)
    if mask.sum() < 2:
        return 0.0
    av, bv = a_np[mask], b_np[mask]
    if av.std() < 1e-8 or bv.std() < 1e-8:
        return 0.0
    try:
        r, _ = pearsonr(av, bv)
        return abs(float(r)) if math.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _rank_normalize(values: list[float]) -> list[float]:
    """將值列表秩歸一化到 [0, 1]（處理 -inf 等哨兵值）。

    -inf / nan 的哨兵值得到 0.0；其餘按秩歸一。
    """
    import numpy as np
    n = len(values)
    if n == 0:
        return []
    arr = np.array(values, dtype=float)
    # 分離哨兵（-inf 或 nan）
    valid_mask = np.isfinite(arr) & (arr > -1e30)
    result = np.zeros(n, dtype=float)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return result.tolist()
    valid_vals = arr[valid_indices]
    # 秩（從小到大，歸一到 [0, 1]）
    order = np.argsort(np.argsort(valid_vals))
    n_valid = len(valid_vals)
    if n_valid == 1:
        result[valid_indices[0]] = 1.0
    else:
        result[valid_indices] = order / (n_valid - 1)
    return result.tolist()


# ── 模組級獨立函數（與 EffectivenessEvaluator 解耦，便於測試）───────────────

def score(
    candidate: torch.Tensor,
    target: torch.Tensor,
    name: str,
    category: str = "",
    w_rankic: float = 0.6,
    w_mi: float = 0.4,
    eval_window: int = 250,
    horizon: int = 1,
) -> ScoreResult:
    """對單個候選打分（R4）。

    單次調用時無跨候選歸一，直接返回原始 ic/rank_ic/mi；
    importance_score 也先存原始 rank_ic（score_all 調用時會覆蓋為歸一後聚合值）。
    """
    # 防 look-ahead 對齊
    cand_a, tgt_a = _align_causal(candidate, target, horizon)

    # 退化檢測
    if _is_degenerate(cand_a):
        return ScoreResult(
            candidate=name,
            category=category,
            ic=0.0,
            rank_ic=0.0,
            ir=0.0,
            mi=0.0,
            importance_score=float("-inf"),
            unscorable=True,
        )

    ic_mean, ric_mean, ir = _compute_ic_rankic(cand_a, tgt_a)
    mi_val = _compute_mi(cand_a, tgt_a)

    # 單次調用時 importance_score 用 |rank_ic| 作為臨時代理
    # score_all 會重算聚合分數
    importance_score = w_rankic * abs(ric_mean) + w_mi * mi_val

    return ScoreResult(
        candidate=name,
        category=category,
        ic=ic_mean,
        rank_ic=ric_mean,
        ir=ir,
        mi=mi_val,
        importance_score=importance_score,
        unscorable=False,
    )


def score_all(
    candidates_dict: dict[str, torch.Tensor],
    target: torch.Tensor,
    categories: Optional[dict[str, str]] = None,
    w_rankic: float = 0.6,
    w_mi: float = 0.4,
    eval_window: int = 250,
    horizon: int = 1,
) -> list[ScoreResult]:
    """對所有候選打分並做跨候選秩歸一，聚合 importance_score（R4.4）。

    退化序列仍保留（unscorable=True，importance_score=-inf），排在最後。
    """
    categories = categories or {}
    raw: list[ScoreResult] = []

    for name, cand in candidates_dict.items():
        cat = categories.get(name, "")
        sr = score(
            candidate=cand,
            target=target,
            name=name,
            category=cat,
            w_rankic=w_rankic,
            w_mi=w_mi,
            eval_window=eval_window,
            horizon=horizon,
        )
        raw.append(sr)

    # 跨候選秩歸一，聚合 importance_score
    abs_rankic = [abs(r.rank_ic) if not r.unscorable else float("-inf") for r in raw]
    mi_vals = [r.mi if not r.unscorable else float("-inf") for r in raw]

    norm_rankic = _rank_normalize(abs_rankic)
    norm_mi = _rank_normalize(mi_vals)

    results: list[ScoreResult] = []
    for i, sr in enumerate(raw):
        if sr.unscorable:
            agg = float("-inf")
        else:
            agg = w_rankic * norm_rankic[i] + w_mi * norm_mi[i]
        results.append(ScoreResult(
            candidate=sr.candidate,
            category=sr.category,
            ic=sr.ic,
            rank_ic=sr.rank_ic,
            ir=sr.ir,
            mi=sr.mi,
            importance_score=agg,
            unscorable=sr.unscorable,
        ))

    return results


def prune(
    scores: list[ScoreResult],
    series_dict: dict[str, torch.Tensor],
    corr_threshold: float = 0.9,
    conservative: bool = False,
    margin: float = 0.01,
) -> list[ReportRow]:
    """保守雙條件相關性剪枝（R5）。

    series_dict: dict[str, Tensor[N,T]]，候選序列（配合 scores 中的名稱）。
    conservative=True 時閾值上調到 0.95。
    margin: 分差閾值，僅當 score_diff > margin 才剪（勢均力敵時保留）。

    返回 list[ReportRow]，含 retention_status 與 pruned_in_favor_of。
    """
    if not (0.0 <= corr_threshold <= 1.0):
        raise ConfigError(
            f"corr_threshold={corr_threshold} 越界，必須在 [0.0, 1.0] 內"
        )

    threshold = 0.95 if conservative else corr_threshold

    # 按 importance_score 降序 + 確定性 tie-break（名稱字母序）
    def sort_key(sr: ScoreResult):
        s = sr.importance_score
        if not math.isfinite(s):
            s = float("-inf")
        return (-s, sr.candidate)

    ordered = sorted(scores, key=sort_key)

    retained: list[ScoreResult] = []   # 已保留
    rows: list[ReportRow] = []

    name_to_series = series_dict

    for sr in ordered:
        pruned_by: Optional[str] = None
        for ret in retained:
            a_series = name_to_series.get(sr.candidate)
            b_series = name_to_series.get(ret.candidate)
            if a_series is None or b_series is None:
                continue
            corr = _pearson_corr(a_series, b_series)
            if corr < threshold:
                continue
            # 相關性超閾值——檢查分差
            s_retained = ret.importance_score
            s_current = sr.importance_score
            if not math.isfinite(s_retained):
                s_retained = float("-inf")
            if not math.isfinite(s_current):
                s_current = float("-inf")
            score_diff = s_retained - s_current
            if score_diff > margin:
                # 確認剪枝
                pruned_by = ret.candidate
                break
            # 分差不夠大（勢均力敵），保留當前候選
        if pruned_by is not None:
            rows.append(ReportRow(
                **{**{f: getattr(sr, f) for f in ScoreResult.__dataclass_fields__},
                   "retention_status": "pruned",
                   "pruned_in_favor_of": pruned_by,
                   "marginal_contribution": None,
                   "drop_recommendation": False},
            ))
        else:
            retained.append(sr)
            rows.append(ReportRow(
                **{**{f: getattr(sr, f) for f in ScoreResult.__dataclass_fields__},
                   "retention_status": "retained",
                   "pruned_in_favor_of": None,
                   "marginal_contribution": None,
                   "drop_recommendation": False},
            ))

    return rows


def _compute_metric(
    candidates_dict: dict[str, torch.Tensor],
    target: torch.Tensor,
    horizon: int,
    w_rankic: float,
    w_mi: float,
) -> float:
    """計算候選集合的整體 metric（|IC| 絕對值均值）。

    供 ablate 內部使用。返回 0.0 若集合為空或全部退化。
    """
    if not candidates_dict:
        return 0.0
    import numpy as np
    ic_vals = []
    for name, cand in candidates_dict.items():
        cand_a, tgt_a = _align_causal(cand, target, horizon)
        if _is_degenerate(cand_a):
            continue
        ic_mean, ric_mean, ir = _compute_ic_rankic(cand_a, tgt_a)
        ic_vals.append(abs(ic_mean))
    if not ic_vals:
        return 0.0
    return float(sum(ic_vals) / len(ic_vals))


def ablate(
    name: str,
    base_set: dict[str, torch.Tensor],
    target: torch.Tensor,
    drop_threshold: float = -0.01,
    n_windows: int = 5,
    horizon: int = 1,
    w_rankic: float = 0.6,
    w_mi: float = 0.4,
) -> AblationResult:
    """單元素消融（R6）。

    base_set 包含被消融項（name 必須存在於 base_set 中）。
    n_windows: 滑動窗口數量（降方差）。
    """
    if name not in base_set:
        return AblationResult(
            name=name,
            marginal_contribution=None,
            drop_recommendation=False,
            error=f"候選 '{name}' 不在 base_set 中",
        )

    without_set = {k: v for k, v in base_set.items() if k != name}

    T_full = target.shape[-1]
    if T_full < 2:
        return AblationResult(
            name=name,
            marginal_contribution=None,
            drop_recommendation=False,
            error="target 序列過短，無法計算 metric",
        )

    # 多窗口滑動（降低單次估計方差）
    window_size = max(2, T_full // n_windows)
    marginals: list[float] = []

    for w_idx in range(n_windows):
        start = w_idx * (T_full // n_windows)
        end = start + window_size
        if end > T_full:
            end = T_full
        if end - start < 2:
            continue

        tgt_w = target[..., start:end]
        with_dict = {k: v[..., start:end] for k, v in base_set.items()}
        without_dict = {k: v[..., start:end] for k, v in without_set.items()}

        try:
            m_with = _compute_metric(with_dict, tgt_w, horizon, w_rankic, w_mi)
        except Exception as e:
            return AblationResult(
                name=name,
                marginal_contribution=None,
                drop_recommendation=False,
                error=f"含候選配置計算失敗（窗口{w_idx}）: {e}",
            )

        try:
            m_without = _compute_metric(without_dict, tgt_w, horizon, w_rankic, w_mi)
        except Exception as e:
            return AblationResult(
                name=name,
                marginal_contribution=None,
                drop_recommendation=False,
                error=f"去候選配置計算失敗（窗口{w_idx}）: {e}",
            )

        marginals.append(m_with - m_without)

    if not marginals:
        return AblationResult(
            name=name,
            marginal_contribution=None,
            drop_recommendation=False,
            error="所有窗口均無法計算 marginal",
        )

    marginal = float(sum(marginals) / len(marginals))
    drop_rec = marginal <= drop_threshold

    return AblationResult(
        name=name,
        marginal_contribution=marginal,
        drop_recommendation=drop_rec,
        error=None,
    )


def build_report(
    rows: list[ReportRow],
    active_subset: list[str],
    vocab_version: str,
    config: dict,
) -> Report:
    """構建排序報告（R7.1）。

    按 importance_score 降序 + 確定性 tie-break（名稱字母序）排序。
    """
    def sort_key(row: ReportRow):
        s = row.importance_score
        if not math.isfinite(s):
            s = float("-inf")
        return (-s, row.candidate)

    sorted_rows = sorted(rows, key=sort_key)
    generated_at = datetime.now(timezone.utc).isoformat()

    return Report(
        vocab_version=vocab_version,
        generated_at=generated_at,
        config=config,
        rows=sorted_rows,
        active_subset=list(active_subset),
    )


def _row_to_dict(row: ReportRow) -> dict:
    """將 ReportRow 序列化為 JSON-ready dict。

    marginal_contribution=None → "not_computed"（R7.2）。
    """
    d = {
        "candidate": row.candidate,
        "category": row.category,
        "ic": row.ic,
        "rank_ic": row.rank_ic,
        "ir": row.ir,
        "mi": row.mi,
        "importance_score": row.importance_score if math.isfinite(row.importance_score) else None,
        "unscorable": row.unscorable,
        "retention_status": row.retention_status,
        "pruned_in_favor_of": row.pruned_in_favor_of,
        "marginal_contribution": (
            "not_computed" if row.marginal_contribution is None
            else row.marginal_contribution
        ),
        "drop_recommendation": row.drop_recommendation,
    }
    return d


def _dict_to_row(d: dict) -> ReportRow:
    """從 JSON dict 反序列化 ReportRow。"not_computed" → marginal_contribution=None。"""
    mc_raw = d.get("marginal_contribution", "not_computed")
    mc: Optional[float] = None if mc_raw == "not_computed" else float(mc_raw)

    imp = d.get("importance_score")
    importance_score = float("-inf") if imp is None else float(imp)

    return ReportRow(
        candidate=d["candidate"],
        category=d.get("category", ""),
        ic=float(d.get("ic", 0.0)),
        rank_ic=float(d.get("rank_ic", 0.0)),
        ir=float(d.get("ir", 0.0)),
        mi=float(d.get("mi", 0.0)),
        importance_score=importance_score,
        unscorable=bool(d.get("unscorable", False)),
        retention_status=d.get("retention_status", "retained"),
        pruned_in_favor_of=d.get("pruned_in_favor_of"),
        marginal_contribution=mc,
        drop_recommendation=bool(d.get("drop_recommendation", False)),
    )


def save_report(report: Report, path: str) -> None:
    """持久化報告到 JSON 文件（R7.5/7.6）。

    先寫臨時文件再原子替換，失敗時保留舊文件並拋 ReportPersistError。
    """
    tmp_path = path + ".tmp"
    data = {
        "vocab_version": report.vocab_version,
        "generated_at": report.generated_at,
        "config": report.config,
        "rows": [_row_to_dict(r) for r in report.rows],
        "active_subset": report.active_subset,
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        # 清理臨時文件（如果存在）
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise ReportPersistError(f"報告持久化失敗，舊文件已保留: {e}") from e


def load_report(path: str) -> Report:
    """從 JSON 文件載入報告（R7.5）。"not_computed" → marginal_contribution=None。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [_dict_to_row(r) for r in data.get("rows", [])]
    return Report(
        vocab_version=data.get("vocab_version", ""),
        generated_at=data.get("generated_at", ""),
        config=data.get("config", {}),
        rows=rows,
        active_subset=data.get("active_subset", []),
    )


def select_active_subset(
    report: Report,
    retention_threshold: float = 0.0,
    max_retained: Optional[int] = None,
) -> list[str]:
    """選擇 Active_Subset（R7.3/7.4/7.7）。

    默認包含語義：threshold=0.0, max_retained=None → 全部未剪枝候選進入。
    篩選是可選的收緊，不是默認破壞性行為。

    空選擇 → Active_Subset 為空，發出 warning（不拋錯，R7.7）。
    """
    # 只取 retained 候選，按 importance_score 降序（tie-break 名稱字母序）
    def sort_key(row: ReportRow):
        s = row.importance_score
        if not math.isfinite(s):
            s = float("-inf")
        return (-s, row.candidate)

    retained_rows = [r for r in report.rows if r.retention_status == "retained"]
    retained_sorted = sorted(retained_rows, key=sort_key)

    # 應用 retention_threshold 篩選
    # unscorable 候選（score=-inf）在默認閾值 0.0 下也保留（保守語義：放過不錯殺）
    # 僅當閾值被顯式收緊（>0.0）或明確啟用篩選時才排除 unscorable
    def _passes_threshold(r: ReportRow) -> bool:
        if r.unscorable:
            # 保守：默認閾值 0.0 下 unscorable 亦保留
            return retention_threshold <= 0.0
        return math.isfinite(r.importance_score) and r.importance_score >= retention_threshold

    filtered = [r for r in retained_sorted if _passes_threshold(r)]

    # 截斷到 max_retained
    if max_retained is not None:
        filtered = filtered[:max_retained]

    if not filtered:
        warnings.warn(
            "select_active_subset: Active_Subset 為空，所有候選均未通過閾值篩選。",
            UserWarning,
            stacklevel=2,
        )

    return [r.candidate for r in filtered]


# ── 主類 EffectivenessEvaluator ─────────────────────────────────────────

class EffectivenessEvaluator:
    """Effectiveness_Evaluator 主類（R4–R7）。

    封裝打分、剪枝、消融、報告構建/持久化與 Active_Subset 選擇。
    所有默認參數均遵循「寧可放過，不要錯殺」的保守原則。

    Parameters
    ----------
    corr_threshold : float
        相關性剪枝閾值，默認 0.9；conservative=True 時上調到 0.95（R5.4）。
    conservative : bool
        保守模式，上調剪枝閾值（R5.4）。
    retention_threshold : float
        Active_Subset 篩選分數下界，默認 0.0（全保留語義，R7.3）。
    max_retained : int | None
        Active_Subset 最大候選數，默認 None（不設上限，R7.3）。
    drop_threshold : float
        消融 drop 建議閾值，帶負值保護帶，默認 -0.01（R6.5）。
    w_rankic : float
        RankIC 權重，默認 0.6（R4.4）。
    w_mi : float
        MI 權重，默認 0.4（R4.4）。
    eval_window : int
        評估窗口大小，默認 250（R4）。
    target_horizon : int
        前視收益 horizon，用於 _align_causal，默認 1（R4.7）。
    """

    def __init__(
        self,
        corr_threshold: float = 0.9,
        conservative: bool = False,
        retention_threshold: float = 0.0,
        max_retained: Optional[int] = None,
        drop_threshold: float = -0.01,
        w_rankic: float = 0.6,
        w_mi: float = 0.4,
        eval_window: int = 250,
        target_horizon: int = 1,
    ) -> None:
        if not (0.0 <= corr_threshold <= 1.0):
            raise ConfigError(
                f"corr_threshold={corr_threshold} 越界，必須在 [0.0, 1.0] 內"
            )
        self.corr_threshold = corr_threshold
        self.conservative = conservative
        self.retention_threshold = retention_threshold
        self.max_retained = max_retained
        self.drop_threshold = drop_threshold
        self.w_rankic = w_rankic
        self.w_mi = w_mi
        self.eval_window = eval_window
        self.target_horizon = target_horizon

    # ── 打分 ────────────────────────────────────────────────────────────

    def score(
        self,
        candidate: torch.Tensor,
        target: torch.Tensor,
        name: str,
        category: str = "",
    ) -> ScoreResult:
        """對單個候選打分（R4）。"""
        return score(
            candidate=candidate,
            target=target,
            name=name,
            category=category,
            w_rankic=self.w_rankic,
            w_mi=self.w_mi,
            eval_window=self.eval_window,
            horizon=self.target_horizon,
        )

    def score_all(
        self,
        candidates: dict[str, torch.Tensor],
        target: torch.Tensor,
        categories: Optional[dict[str, str]] = None,
    ) -> list[ScoreResult]:
        """對所有候選打分並做跨候選秩歸一（R4）。"""
        return score_all(
            candidates_dict=candidates,
            target=target,
            categories=categories,
            w_rankic=self.w_rankic,
            w_mi=self.w_mi,
            eval_window=self.eval_window,
            horizon=self.target_horizon,
        )

    # ── 剪枝 ────────────────────────────────────────────────────────────

    def prune(
        self,
        scores: list[ScoreResult],
        series: dict[str, torch.Tensor],
    ) -> list[ReportRow]:
        """保守雙條件相關性剪枝（R5）。"""
        return prune(
            scores=scores,
            series_dict=series,
            corr_threshold=self.corr_threshold,
            conservative=self.conservative,
        )

    # ── 消融 ────────────────────────────────────────────────────────────

    def ablate(
        self,
        name: str,
        base_set: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> AblationResult:
        """單元素消融（R6）。"""
        return ablate(
            name=name,
            base_set=base_set,
            target=target,
            drop_threshold=self.drop_threshold,
            horizon=self.target_horizon,
            w_rankic=self.w_rankic,
            w_mi=self.w_mi,
        )

    # ── 報告 ────────────────────────────────────────────────────────────

    def build_report(
        self,
        rows: list[ReportRow],
        active_subset: list[str],
    ) -> Report:
        """構建排序報告（R7.1）。"""
        try:
            from .vocab import VOCAB_VERSION
        except Exception:
            VOCAB_VERSION = "unknown"
        config_snapshot = {
            "corr_threshold": self.corr_threshold,
            "conservative": self.conservative,
            "retention_threshold": self.retention_threshold,
            "max_retained": self.max_retained,
            "drop_threshold": self.drop_threshold,
            "w_rankic": self.w_rankic,
            "w_mi": self.w_mi,
            "eval_window": self.eval_window,
            "target_horizon": self.target_horizon,
        }
        return build_report(
            rows=rows,
            active_subset=active_subset,
            vocab_version=VOCAB_VERSION,
            config=config_snapshot,
        )

    def save_report(self, report: Report, path: str) -> None:
        """持久化報告（R7.5/7.6）。"""
        save_report(report, path)

    @staticmethod
    def load_report(path: str) -> Report:
        """載入報告（R7.5）。"""
        return load_report(path)

    # ── Active_Subset ───────────────────────────────────────────────────

    def select_active_subset(self, report: Report) -> list[str]:
        """選擇 Active_Subset（R7.3/7.4/7.7）。默認全保留。"""
        return select_active_subset(
            report=report,
            retention_threshold=self.retention_threshold,
            max_retained=self.max_retained,
        )

    # ── 靜態工具 ────────────────────────────────────────────────────────

    @staticmethod
    def _align_causal(
        candidate: torch.Tensor,
        target: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """防 look-ahead 的時間對齊（R4.7）。"""
        return _align_causal(candidate, target, horizon)
