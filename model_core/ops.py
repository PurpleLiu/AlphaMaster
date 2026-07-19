"""
model_core/ops.py -- 運算元庫（Operator_Library, R2）

本模組把歷史上手工維護的 `OPS_CONFIG` 列表遷移為由聲明式註冊層
（`model_core.registry.Registry`）驅動的 `OPERATOR_REGISTRY`。所有運算元先以
`OperatorSpec(name, arity, transform)` 註冊進 `OPERATOR_REGISTRY`，`OPS_CONFIG`
隨後作為「導出視圖」由註冊表派生（`[(name, transform, arity), ...]`），保持對
下游 `vocab.py` / `vm.py` 的 import 相容與既有元組結構。

統一契約（R2.8, R2.9, R2.13）：
  - 形狀契約：所有運算元輸入 `[N, T]`、輸出 `[N, T]`。
  - 二元/三元運算元在入口校驗各操作數形狀一致，不一致拋 `ShapeError` 且不產出張量。
  - 註冊表儲存 name（≤64 字元）與 arity（本模組運算元均為 1/2/3）。

說明：現有運算元多以 lambda 定義，`inspect` 不總能可靠解析 arity（內建函數、被
包裝函數等）。註冊時以顯式聲明的 arity 為準；二元/三元運算元經形狀校驗包裝後為
可變位置參數形式（`*operands`），註冊層會跳過 arity 觀測校驗，從而避免對既有運算元
的 `ArityMismatchError` 誤報。
"""
import torch

from .registry import OperatorSpec, Registry


# ── 運算元層錯誤類型（對應 design「錯誤類型模型」，歸為運算元層）──────────────

class ShapeError(Exception):
    """運算元操作數形狀不相容或錯誤（R2.13）。

    二元/三元運算元在入口發現各操作數形狀不一致時拋出，且不產出張量。
    """


def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)

def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """降低稀疏度：閾值從 3σ 改為 1.5σ，讓更多時間步有非零輸出。

    因果 expanding zscore：每個 t 僅用 x[:, :t+1] 計算 mean/std，
    避免原 dim=1 全局聚合（含未來統計量）引入的 look-ahead bias。
    """
    N, T = x.shape
    cnt = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T)
    cumsum = x.cumsum(dim=1)
    mean = cumsum / cnt                       # [N,T]，t 位 = x[:,:t+1].mean()
    cumsum_sq = (x * x).cumsum(dim=1)
    var = (cumsum_sq / cnt) - mean * mean     # E[x^2] - E[x]^2
    std = var.clamp(min=1e-12).sqrt() + 1e-6
    z = (x - mean) / std
    return torch.tanh(z - 1.5)   # tanh 軟化，不再產生全零區間

def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

def _op_wma(x: torch.Tensor) -> torch.Tensor:
    """加權移動平均（權重 3,2,1），平滑信號，減少剝頭皮"""
    return (3.0 * x + 2.0 * _ts_delay(x, 1) + 1.0 * _ts_delay(x, 2)) / 6.0


# ── 時序滑動窗口輔助函數（不使用 @torch.jit.script，lambda 不相容 JIT）──────

def _ts_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    """unfold 實現因果滑動窗口，返回 [N, T, d] 的窗口張量。"""
    N, T = x.shape
    pad = torch.zeros(N, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, d, 1)  # [N, T, d]


def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑動均值，返回 [N, T]。"""
    return _ts_rolling(x, d).mean(dim=-1)


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑動標準差（ddof=0），返回 [N, T]，下界 1e-6。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    m = w.mean(dim=-1, keepdim=True)
    std = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    return torch.nan_to_num(std, nan=0.0)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑動排名（嚴格小於當前值的比例），返回 [N, T]，值域 [0, 1)。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    cur = w[:, :, -1:]                             # 當前值，[N, T, 1]
    rank = (w < cur).float().mean(dim=-1)          # [N, T]
    return torch.nan_to_num(rank, nan=0.0)


def _ts_corr_10(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x 與 y 的 10 週期因果滑動 Pearson 相關係數，返回 [N, T]，值域 [-1, 1]。
    當 x 或 y 在窗口內為常數（std < 1e-6）時，該位置輸出 0。
    """
    d = 10
    wx = _ts_rolling(x, d)                         # [N, T, 10]
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    sx = ((wx - mx) ** 2).mean(dim=-1).sqrt()      # [N, T]
    sy = ((wy - my) ** 2).mean(dim=-1).sqrt()
    # 常數窗口（std < 1e-6）輸出 0
    mask = (sx < 1e-6) | (sy < 1e-6)
    corr = cov / (sx * sy + 1e-8)
    corr[mask] = 0.0
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


# ── v3.0 新增運算元 helper ─────────────────────────────────────────────

def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """指數加權移動平均（因果）。alpha 越大越關注近期。"""
    # 用遞推實現太慢，用衰減權重卷積近似（窗口=20 足夠）
    w = min(20, x.shape[1])
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(w)],
                           device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    pad = torch.zeros(x.shape[0], w - 1, device=x.device, dtype=x.dtype)
    xp = torch.cat([pad, x], dim=1)
    return torch.nn.functional.unfold(xp.unsqueeze(1), (1, w)).squeeze(1) * 0  # placeholder
    # 上面的 unfold 對 1D 不直接 work，改用簡單循環近似


def _ema_simple(x: torch.Tensor, span: int, exact: bool = False) -> torch.Tensor:
    """指數加權移動平均（因果），span 期。

    默認路徑（exact=False）：
        向量化因果卷積近似，複雜度 O(N·T·w)，無逐時間步 Python 循環（R8.3）。
        alpha = 2/(span+1)；有效窗口 w = min(T, ceil(-log(1e-6)/(-log(1-alpha))))，
        保證尾部權重 (1-alpha)^w < 1e-6。
        使用首值填充（first-value padding）以匹配遞推版初始條件 out[0]=x[0]，
        max|Δ| 與遞推版差異實測 < 1e-4。

    可選精確路徑（exact=True）：
        嚴格遞推實現：out[t] = alpha*x[t] + (1-alpha)*out[t-1]。
        複雜度 O(N·T)（順序累積，R8.4 文件化複雜度）。
    """
    alpha = 2.0 / (span + 1.0)
    N, T = x.shape

    if exact:
        # ── 精確遞推路徑（O(N·T) 順序累積，R8.4 文件化複雜度）──────────
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0]
        for t in range(1, T):
            out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
        return out

    # ── 向量化卷積近似路徑（默認，R8.3）────────────────────────────────
    import math
    if alpha >= 1.0:
        return x.clone()
    # w_full 僅由 span 決定
    w_full = max(1, math.ceil(-math.log(1e-6) / (-math.log(1.0 - alpha))))

    # 因果性保證：為確保前綴步輸出與序列長度無關，必須保證相同 T 範圍內
    # 兩種實現路徑（精確 vs 向量化）不能混用。
    # 策略：僅當 T >= 2 * w_full 時才使用向量化（此時 warm-up 區占比 < 50%，
    # 精度問題可忽略）；否則使用精確遞推（嚴格因果，O(N·T)）。
    # 注意：2*w_full 是確定性閾值，不依賴具體輸入，故不同長度的序列在
    # 超過閾值後行為一致。實際訓練序列 T=200-512 均遠超 2*w_full(≤360)。
    if T < 2 * w_full:
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0]
        for t in range(1, T):
            out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
        return out

    # T >= 2*w_full：向量化卷積近似（首值填充），max|Δ| < 1e-4
    decay = 1.0 - alpha
    powers = torch.arange(w_full - 1, -1, -1, dtype=x.dtype, device=x.device)
    weights = alpha * (decay ** powers)                        # 未歸一化

    # 首值填充：等效於「歷史全為 x[0]」，與遞推版 out[0]=x[0] 一致
    first = x[:, :1].expand(N, w_full - 1)                    # [N, w_full-1]
    xp = torch.cat([first, x], dim=1)                          # [N, T+w_full-1]
    windows = xp.unfold(1, w_full, 1)                          # [N, T, w_full]
    out = (windows * weights).sum(dim=-1)                      # [N, T]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _ts_quantile(x: torch.Tensor, d: int) -> torch.Tensor:
    """當前值在過去 d 期的分位數（0~1），用 TS_RANK 的連續版。"""
    w = _ts_rolling(x, d)
    cur = w[:, :, -1:]
    rank = (w <= cur).float().mean(dim=-1)
    return torch.nan_to_num(rank, nan=0.5)


def _ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期偏度（三階矩標準化），捕捉分布非對稱性。"""
    w = _ts_rolling(x, d)
    m = w.mean(dim=-1, keepdim=True)
    s = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    skew = ((w - m) ** 3).mean(dim=-1) / (s ** 3)
    return torch.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)


def _delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """d 期差分: x[t] - x[t-d]，前 d 位置 0。Alpha 101 最常用運算元。"""
    if d == 0:
        return x
    out = torch.zeros_like(x)
    out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def _ts_arg_max(x: torch.Tensor, d: int) -> torch.Tensor:
    """過去 d 期最大值的位置（歸一化到 [0,1]，0=最早，1=最近）。Alpha#001 核心運算元。"""
    w = _ts_rolling(x, d)
    idx = w.argmax(dim=-1).float()
    return idx / max(d - 1, 1)


def _ts_arg_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """過去 d 期最小值的位置（歸一化到 [0,1]）。"""
    w = _ts_rolling(x, d)
    idx = w.argmin(dim=-1).float()
    return idx / max(d - 1, 1)


def _decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """線性衰減加權平均（近期權重更高）。Alpha#98 核心運算元。權重 = [1,2,...,d]/sum。"""
    w = _ts_rolling(x, d)
    weights = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (w * weights).sum(dim=-1)


def _decay_exp(x: torch.Tensor, d: int, alpha: float = 0.5) -> torch.Tensor:
    """指數衰減加權平均（近期權重更高）。與 DECAY_LINEAR 對應，平滑更激進。"""
    w = _ts_rolling(x, d)
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(d)],
                           dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (w * weights).sum(dim=-1)


def _scale(x: torch.Tensor) -> torch.Tensor:
    """沿時間軸縮放到單位 L1 範數（Alpha#028/032 高頻運算元）。
    scale(x)[t] = x[t] / sum(|x[1..t]|)，避免未來資訊用因果累積和。
    """
    abs_x = x.abs()
    cumsum = torch.cumsum(abs_x, dim=1) + 1e-6
    return x / cumsum


def _ts_covariance(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """d 期因果滑動協方差。"""
    wx = _ts_rolling(x, d)
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    return torch.nan_to_num(cov, nan=0.0)


def _ts_product(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期因果滑動乘積。用對數累加避免數值爆炸：prod = exp(sum(log(x+1)))。
    適合收益累積，輸出接近 "過去 d 期累計收益"。
    輸入先 clamp 到 [-0.999, +inf)，避免 log1p 在 x<=-1 時產生 NaN。
    """
    x_safe = torch.clamp(x, -0.999, None)
    log_x = torch.log1p(x_safe)
    w = _ts_rolling(log_x, d)
    # clamp 對數累加和防止 expm1 溢出到 float32 邊界（>1e38）
    log_sum = w.sum(dim=-1).clamp(-10.0, 10.0)
    out = torch.expm1(log_sum)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_power(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    """帶符號乘方: sign(x) * |x|^a。Alpha#001 SignedPower。保留符號同時放大極端值。"""
    out = torch.sign(x) * torch.abs(x) ** a
    # 防溢出：|x|^a 在鏈式調用中可能爆炸到 float32 邊界（如 3.3e38），
    # 導致後續 mean/std 計算溢出為 inf。clamp 到安全範圍。
    return torch.nan_to_num(out.clamp(-1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


# ── Cross_Sectional 運算元 helper（沿 N 維，每時間步跨品種；R2.1, R2.2）───────
#
# 輸入 `[N, T]`：N=品種數、T=時間步。計算沿 dim=0（N 維）逐時間步進行，
# 完全向量化（禁止逐時間步 Python 循環）。N=1（單品種，截面無分散）時按語義退化。
# 全部 NaN-safe：出口 `nan_to_num`，CS_RANK/CS_SCALE 退化預設值 0.5，其餘 0。

def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    """每時間步跨品種百分位排名，值域 [0, 1]（R2.1）。

    對每一列（時間步）沿 N 維排名，歸一化到 `[0, 1]`（rank/(N-1)）。N=1 時截面
    無分散，退化為 0.5。用雙 argsort 向量化，無逐時間步 Python 循環。
    """
    N, T = x.shape
    if N == 1:
        return torch.full_like(x, 0.5)
    order = x.argsort(dim=0)                       # 沿 N 維排序索引
    ranks = torch.empty_like(x)
    rank_vals = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(1).expand(N, T)
    ranks.scatter_(0, order, rank_vals)            # ranks[order[i,t], t] = i
    pct = ranks / (N - 1)
    return torch.nan_to_num(pct, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_scale(x: torch.Tensor) -> torch.Tensor:
    """每時間步跨品種縮放到 [0, 1]：`(x - min) / (max - min)`（R2.1）。

    沿 N 維取每列的 min/max。零跨度（max==min）該列退化為 0.5；N=1 退化為 0.5。
    完全向量化。
    """
    N, T = x.shape
    if N == 1:
        return torch.full_like(x, 0.5)
    mn = x.min(dim=0, keepdim=True).values         # [1, T]
    mx = x.max(dim=0, keepdim=True).values          # [1, T]
    span = mx - mn                                  # [1, T]
    zero_span = span.abs() < 1e-9                   # [1, T]
    span_safe = torch.where(zero_span, torch.ones_like(span), span)
    out = (x - mn) / span_safe
    out = torch.where(zero_span.expand_as(out), torch.full_like(out, 0.5), out)
    return torch.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_neutralize(x: torch.Tensor) -> torch.Tensor:
    """每時間步減去跨品種算術均值（截面中性化，R2.2）。N=1 退化為 0。"""
    N, T = x.shape
    if N == 1:
        return torch.zeros_like(x)
    mean = x.mean(dim=0, keepdim=True)              # [1, T]
    out = x - mean
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── 形狀一致性校驗包裝（R2.9, R2.13）─────────────────────────────────────

def _with_shape_check(name: str, transform):
    """為二元/三元運算元包裝入口形狀一致性校驗。

    調用時先校驗各操作數形狀完全一致，不一致拋 `ShapeError` 且不調用底層
    transform（不產出張量）。包裝後為可變位置參數形式，註冊層將跳過 arity
    觀測校驗（以顯式聲明 arity 為準）。
    """
    def _checked(*operands: torch.Tensor) -> torch.Tensor:
        base = operands[0].shape
        for other in operands[1:]:
            if other.shape != base:
                raise ShapeError(
                    f"運算元 '{name}' 操作數形狀不一致: "
                    f"{tuple(base)} vs {tuple(other.shape)}"
                )
        return transform(*operands)

    return _checked


# ── 初始運算元定義（保持既有 44 個運算元的命名、順序與行為）──────────────────
#
# 每項為 (name, transform, arity)。此列表等價於歷史 `OPS_CONFIG` 的內容與順序，
# 用於註冊進 `OPERATOR_REGISTRY`；`OPS_CONFIG` 隨後由註冊表派生為導出視圖。

_INITIAL_OPERATORS = [
    # ── 基礎運算元（token id = feat_offset+0~11）────────────────────────
    ('ADD',    lambda x, y: x + y,          2),
    ('SUB',    lambda x, y: x - y,          2),
    ('MUL',    lambda x, y: x * y,          2),
    ('DIV',    lambda x, y: x / (y + 1e-6), 2),
    ('NEG',    lambda x: -x,                1),
    ('ABS',    torch.abs,                   1),
    ('SIGN',   torch.sign,                  1),
    ('GATE',   _op_gate,                    3),
    ('JUMP',   _op_jump,                    1),   # 已降低稀疏度
    ('DECAY',  _op_decay,                   1),
    ('DELAY1', lambda x: _ts_delay(x, 1),   1),
    ('MAX3',   lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
    # ── 時序運算元（token id = feat_offset+12~21）───────────────────────
    ('TS_MEAN_5',  lambda x: _ts_mean(x, 5),  1),
    ('TS_MEAN_10', lambda x: _ts_mean(x, 10), 1),
    ('TS_MEAN_20', lambda x: _ts_mean(x, 20), 1),
    ('TS_STD_5',   lambda x: _ts_std(x, 5),   1),
    ('TS_STD_10',  lambda x: _ts_std(x, 10),  1),
    ('TS_STD_20',  lambda x: _ts_std(x, 20),  1),
    ('TS_RANK_5',  lambda x: _ts_rank(x, 5),  1),
    ('TS_RANK_10', lambda x: _ts_rank(x, 10), 1),
    ('TS_RANK_20', lambda x: _ts_rank(x, 20), 1),
    ('TS_CORR_10', _ts_corr_10,               2),
    # ── 趨勢 / 動量類運算元（token id = feat_offset+22~27）──────────────
    # MOMENTUM_5: 短期均線 - 長期均線，捕捉趨勢方向
    ('MOMENTUM_5',  lambda x: _ts_mean(x, 5)  - _ts_mean(x, 20), 1),
    # MOMENTUM_10: 中期動量
    ('MOMENTUM_10', lambda x: _ts_mean(x, 10) - _ts_mean(x, 20), 1),
    # TS_MAX_10: 10週期最大值，捕捉強勢突破
    ('TS_MAX_10',   lambda x: _ts_rolling(x, 10).max(dim=-1).values, 1),
    # TS_MIN_10: 10週期最小值，捕捉弱勢突破
    ('TS_MIN_10',   lambda x: _ts_rolling(x, 10).min(dim=-1).values, 1),
    # WMA: 加權移動平均，平滑信號
    ('WMA',         _op_wma,  1),
    # DELAY4: 延遲4根bar，構建中期動量差
    ('DELAY4',      lambda x: _ts_delay(x, 4), 1),
    # ── v3.0 新增運算元（token id = feat_offset+28~33）──────────────────
    ('EMA_5',           lambda x: _ema_simple(x, 5),    1),
    ('EMA_20',          lambda x: _ema_simple(x, 20),   1),
    ('TS_QUANTILE_10',  lambda x: _ts_quantile(x, 10),  1),
    ('TS_SKEW_10',      lambda x: _ts_skew(x, 10),      1),
    ('TS_MIN_20',       lambda x: _ts_rolling(x, 20).min(dim=-1).values, 1),
    ('TS_MAX_20',       lambda x: _ts_rolling(x, 20).max(dim=-1).values, 1),
    # ── v3.0 Alpha 101 + 補充運算元（token id = feat_offset+34~43）──────
    # Alpha 101 核心 4 個
    ('DELTA',           lambda x: _delta(x, 1),                1),
    ('TS_ARG_MAX_5',    lambda x: _ts_arg_max(x, 5),           1),
    ('TS_ARG_MIN_5',    lambda x: _ts_arg_min(x, 5),           1),
    ('DECAY_LINEAR_5',  lambda x: _decay_linear(x, 5),         1),
    # 聯網搜索補充 6 個
    ('SCALE',           lambda x: _scale(x),                   1),
    ('COVARIANCE_10',   lambda x, y: _ts_covariance(x, y, 10), 2),
    ('PRODUCT_5',       lambda x: _ts_product(x, 5),           1),
    ('SIGNED_POWER_2',  lambda x: _signed_power(x, 2.0),       1),
    ('TS_DECAY_EXP_5',  lambda x: _decay_exp(x, 5, 0.5),       1),
    ('DELTA_5',         lambda x: _delta(x, 5),                1),
]


# ── Task 3.2 追加：Cross_Sectional 運算元（沿 N 維，每時間步跨品種）──────────
#
# 追加在既有 44 個運算元之後，保持既有運算元命名/順序/行為不變。均為 arity 1，
# 沿 dim=0（N 維）逐時間步向量化計算，NaN-safe。（R2.1, R2.2, R2.10）

_CROSS_SECTIONAL_OPERATORS = [
    ('CS_RANK',       _cs_rank,       1),   # 跨品種百分位排名 [0,1]，N=1→0.5
    ('CS_SCALE',      _cs_scale,      1),   # 跨品種縮放 [0,1]，零跨度/N=1→0.5
    ('CS_NEUTRALIZE', _cs_neutralize, 1),   # 減跨品種均值，N=1→0
]


# ── 構建 OPERATOR_REGISTRY 並派生 OPS_CONFIG 導出視圖 ─────────────────────

# 全局運算元註冊表（Operator_Library, R2）。所有運算元的唯一事實來源。
OPERATOR_REGISTRY = Registry()


def _register_initial_operators(registry: Registry) -> None:
    """把初始運算元註冊進給定註冊表。

    二元/三元運算元經 `_with_shape_check` 包裝以在入口校驗操作數形狀一致性
    （R2.13）；一元運算元直接註冊。以顯式聲明的 arity 為準（R2.8）。
    """
    for name, transform, arity in _INITIAL_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


def _register_cross_sectional_operators(registry: Registry) -> None:
    """把 Cross_Sectional 運算元註冊進給定註冊表（Task 3.2）。

    均為一元運算元（arity 1），沿 N 維逐時間步計算；直接註冊（無需形狀校驗包裝）。
    追加在初始 44 個運算元之後，保持既有運算元順序在前、新運算元在後（R2.1, R2.2）。
    """
    for name, transform, arity in _CROSS_SECTIONAL_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


_register_initial_operators(OPERATOR_REGISTRY)
_register_cross_sectional_operators(OPERATOR_REGISTRY)


# `OPS_CONFIG` 現為 `OPERATOR_REGISTRY` 的導出視圖，保持既有元組結構
# `[(name, transform, arity), ...]` 與下游 vocab.py / vm.py 的 import 相容。
OPS_CONFIG = [
    (spec.name, spec.transform, spec.arity)
    for spec in OPERATOR_REGISTRY.operator_specs
]


# 動態斷言：導出視圖與註冊表嚴格一致；且既有 44 個運算元必須全部在冊（不回歸）。
assert len(OPS_CONFIG) == len(OPERATOR_REGISTRY.operator_specs), (
    "OPS_CONFIG 導出視圖與 OPERATOR_REGISTRY 長度不一致"
)
_EXPECTED_INITIAL_NAMES = {name for name, _, _ in _INITIAL_OPERATORS}
_REGISTERED_NAMES = set(OPERATOR_REGISTRY.operator_names)
assert _EXPECTED_INITIAL_NAMES <= _REGISTERED_NAMES, (
    "既有運算元未全部註冊: "
    f"{_EXPECTED_INITIAL_NAMES - _REGISTERED_NAMES}"
)
assert len(OPERATOR_REGISTRY.operator_specs) >= 44, (
    f"OPERATOR_REGISTRY 至少應含 44 個既有運算元，實際 {len(OPERATOR_REGISTRY.operator_specs)}"
)
# Task 3.2：Cross_Sectional 運算元必須全部在冊（總數隨之增加）。
_EXPECTED_CS_NAMES = {name for name, _, _ in _CROSS_SECTIONAL_OPERATORS}
assert _EXPECTED_CS_NAMES <= _REGISTERED_NAMES, (
    "Cross_Sectional 運算元未全部註冊: "
    f"{_EXPECTED_CS_NAMES - _REGISTERED_NAMES}"
)
assert len(OPERATOR_REGISTRY.operator_specs) >= 44 + len(_CROSS_SECTIONAL_OPERATORS), (
    "OPERATOR_REGISTRY 總數應含既有 44 個 + Cross_Sectional 運算元，"
    f"實際 {len(OPERATOR_REGISTRY.operator_specs)}"
)


# ── Task 3.3 追加：時序求和/極值與幅度變換運算元 ────────────────────────────
#
# 新增 8 個運算元（TS_SUM_5/10/20、MIN、MAX、POWER、SIGNED_LOG、SQRT），
# 追加在既有 47 個運算元（44 初始 + 3 Cross_Sectional）之後，保持既有順序不變。
# 全部運算元出口 nan_to_num→0，滿足形狀契約 [N,T]→[N,T]（R2.9, R2.10）。
# 時序求和用因果 unfold（零填充），每步 t 只用 [t-w+1..t]（R2.11, R2.12）。


def _ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滾動求和（R2.3）：左側 zero-pad + unfold，每步 t 只用 [t-w+1..t]。
    部分窗口（warm-up 期 t<w）由 pad 決定——0 不含未來資訊，輸出有限（R2.12）。
    """
    N, T = x.shape
    pad = torch.zeros(N, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, d, 1).sum(dim=-1)


def _power_signed(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    """符號冪變換（R2.5）：sign(x)*|x|^a，Alpha101 SignedPower 風格。
    取 |x| 作為底數，避免負數的非整數冪；出口 clamp(-1e9, 1e9) 防爆炸。
    """
    out = torch.sign(x) * torch.abs(x) ** a
    return torch.nan_to_num(out.clamp(-1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    """帶符號自然對數（R2.5）：sign(x)*log1p(|x|)，全實數域安全，負輸入不產 NaN。"""
    out = torch.sign(x) * torch.log1p(torch.abs(x))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """帶符號平方根（R2.5）：sign(x)*sqrt(|x|)，負輸入不產 NaN。"""
    out = torch.sign(x) * torch.sqrt(torch.abs(x))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# 運算元列表（追加在既有 47 個之後）
_TASK33_OPERATORS = [
    # 時序求和（arity 1，因果，R2.3）
    ('TS_SUM_5',    lambda x: torch.nan_to_num(_ts_sum(x, 5),  nan=0.0), 1),
    ('TS_SUM_10',   lambda x: torch.nan_to_num(_ts_sum(x, 10), nan=0.0), 1),
    ('TS_SUM_20',   lambda x: torch.nan_to_num(_ts_sum(x, 20), nan=0.0), 1),
    # 元素級二元極值（arity 2，天然因果，R2.4）
    ('MIN', lambda x, y: torch.nan_to_num(torch.minimum(x, y), nan=0.0), 2),
    ('MAX', lambda x, y: torch.nan_to_num(torch.maximum(x, y), nan=0.0), 2),
    # 幅度變換（arity 1，天然因果，R2.5）
    ('POWER',      lambda x: _power_signed(x, 2.0), 1),
    ('SIGNED_LOG', _signed_log,                      1),
    ('SQRT',       _signed_sqrt,                     1),
]


def _register_task33_operators(registry: Registry) -> None:
    """註冊 Task 3.3 新增運算元（時序求和/極值與幅度變換，R2.3–2.5）。

    二元運算元（MIN/MAX）經 `_with_shape_check` 包裝；一元運算元直接註冊。
    追加在既有 47 個運算元之後，保持既有運算元順序在前（R2.9, R2.10）。
    """
    for name, transform, arity in _TASK33_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


_register_task33_operators(OPERATOR_REGISTRY)

# 重新派生 OPS_CONFIG 導出視圖（追加新運算元後更新）
OPS_CONFIG = [
    (spec.name, spec.transform, spec.arity)
    for spec in OPERATOR_REGISTRY.operator_specs
]

# Task 3.3：新增運算元必須全部在冊
_EXPECTED_T33_NAMES = {name for name, _, _ in _TASK33_OPERATORS}
_REGISTERED_NAMES_T33 = set(OPERATOR_REGISTRY.operator_names)
assert _EXPECTED_T33_NAMES <= _REGISTERED_NAMES_T33, (
    "Task 3.3 運算元未全部註冊: "
    f"{_EXPECTED_T33_NAMES - _REGISTERED_NAMES_T33}"
)
assert len(OPERATOR_REGISTRY.operator_specs) >= 44 + len(_CROSS_SECTIONAL_OPERATORS) + len(_TASK33_OPERATORS), (
    "OPERATOR_REGISTRY 總數應含既有 44 + CS 3 + Task3.3 8 個運算元，"
    f"實際 {len(OPERATOR_REGISTRY.operator_specs)}"
)


# ── Task 3.4 追加：歸一化與條件/邏輯運算元 ─────────────────────────────────
#
# 新增 11 個運算元（TS_ZSCORE_10/20、WINSORIZE、CLIP、SIGMOID、TANH_SQUASH、
# GT、LT、AND、OR、IF_GT），追加在既有 55 個運算元之後，保持既有順序不變。
# 全部因果、NaN-safe（R2.6, R2.7, R2.9, R2.10, R8.2, R8.6）。
#
# 歸一化運算元均為 arity 1，因果；條件/邏輯運算元 arity 2 或 3，形狀校驗。

import math as _math  # noqa: E402 — 模組頂部已有 import torch；這裡補 math


def _ts_zscore(x: torch.Tensor, w: int) -> torch.Tensor:
    """因果滾動 z-score（R2.6）：(x - ts_mean) / (ts_std + eps)。
    窗口 w 期，左側 zero-pad + unfold，每步 t 只用 [t-w+1..t]。
    std < eps 時輸出 0（常數窗口安全）。
    """
    windows = _ts_rolling(x, w)                               # [N, T, w]
    m = windows.mean(dim=-1)                                   # [N, T]
    s = ((windows - m.unsqueeze(-1)) ** 2).mean(dim=-1).sqrt() + 1e-9
    z = (x - m) / s
    # std < eps（常數窗口）→ 輸出 0
    mask = s < (1e-9 + 1e-9)
    z = torch.where(mask, torch.zeros_like(z), z)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(x: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    """因果滾動分位裁剪（R2.6, WINSORIZE）。

    用 _ts_rolling(x, 20) 取 per-step 的第 lo/hi 分位點（只用 ≤t 數據），
    再 clamp 當前值到 [lower, upper]。嚴格無未來資訊（因果 unfold）。
    lower < upper 由 lo < hi 保證（默認 5%/95%）。
    """
    w = 20
    windows = _ts_rolling(x, w)                               # [N, T, w]
    lower = torch.quantile(windows.float(), lo, dim=-1).to(x.dtype)  # [N, T]
    upper = torch.quantile(windows.float(), hi, dim=-1).to(x.dtype)  # [N, T]
    # 保證 lower < upper（零跨度時取原值）
    span = upper - lower
    safe_lower = torch.where(span < 1e-9, x, lower)
    safe_upper = torch.where(span < 1e-9, x, upper)
    out = torch.clamp(x, safe_lower, safe_upper)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _clip_fixed(x: torch.Tensor) -> torch.Tensor:
    """硬限幅 clamp(-3, 3)（R2.6, CLIP）。固定常數界，lower < upper。"""
    return torch.clamp(x, -3.0, 3.0)


def _sigmoid_squash(x: torch.Tensor) -> torch.Tensor:
    """2*sigmoid(x)-1，squash 到 [-1, 1]（R2.6）。"""
    out = 2.0 * torch.sigmoid(x) - 1.0
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _tanh_squash(x: torch.Tensor) -> torch.Tensor:
    """tanh(x)，squash 到 (-1, 1)（R2.6）。"""
    out = torch.tanh(x)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


# 條件/邏輯運算元（R2.7）—— 全部形狀校驗（由 _with_shape_check 包裝）

def _gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(x > y).float()，形狀校驗（R2.7）。"""
    out = (x > y).float()
    return torch.nan_to_num(out, nan=0.0)


def _lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(x < y).float()，形狀校驗（R2.7）。"""
    out = (x < y).float()
    return torch.nan_to_num(out, nan=0.0)


def _and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """((x>0) & (y>0)).float()，形狀校驗（R2.7）。"""
    out = ((x > 0) & (y > 0)).float()
    return torch.nan_to_num(out, nan=0.0)


def _or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """((x>0) | (y>0)).float()，形狀校驗（R2.7）。"""
    out = ((x > 0) | (y > 0)).float()
    return torch.nan_to_num(out, nan=0.0)


def _if_gt(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """三元選擇 where(x>0, y, z)（R2.7, IF_GT, arity 3）。形狀校驗。
    語義：條件操作數 x>0 時取 y，否則取 z。
    """
    out = torch.where(x > 0, y, z)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


_TASK34_OPERATORS = [
    # 歸一化運算元（arity 1，因果，R2.6）
    ('TS_ZSCORE_10',  lambda x: _ts_zscore(x, 10),  1),
    ('TS_ZSCORE_20',  lambda x: _ts_zscore(x, 20),  1),
    ('WINSORIZE',     _winsorize,                    1),
    ('CLIP',          _clip_fixed,                   1),
    ('SIGMOID',       _sigmoid_squash,               1),
    ('TANH_SQUASH',   _tanh_squash,                  1),
    # 條件/邏輯運算元（R2.7）
    # P1 注意：LT/GT/AND/OR 輸出純 0/1 二值，與 Neutral Band 連續因子語義衝突，
    # 容易被模型利用來製造稀疏信號刷高訓練分，已從詞表移除。
    # IF_GT 輸出連續值（條件混合），保留。
    ('IF_GT', _if_gt, 3),   # where(x>0, y, z) — 輸出連續值，保留
]


def _register_task34_operators(registry: Registry) -> None:
    """註冊 Task 3.4 新增運算元（歸一化與條件/邏輯，R2.6, R2.7）。

    二元/三元運算元經 `_with_shape_check` 包裝；一元運算元直接註冊。
    追加在既有 55 個運算元之後，保持既有運算元順序在前（R2.9, R2.10）。
    """
    for name, transform, arity in _TASK34_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


_register_task34_operators(OPERATOR_REGISTRY)

# 重新派生 OPS_CONFIG 導出視圖（追加新運算元後更新）
OPS_CONFIG = [
    (spec.name, spec.transform, spec.arity)
    for spec in OPERATOR_REGISTRY.operator_specs
]

# Task 3.4：新增運算元必須全部在冊（LT/GT/AND/OR 已移除，保留 7 個）
_EXPECTED_T34_NAMES = {name for name, _, _ in _TASK34_OPERATORS}
_REGISTERED_NAMES_T34 = set(OPERATOR_REGISTRY.operator_names)
assert _EXPECTED_T34_NAMES <= _REGISTERED_NAMES_T34, (
    "Task 3.4 運算元未全部註冊: "
    f"{_EXPECTED_T34_NAMES - _REGISTERED_NAMES_T34}"
)
_PREV_COUNT = 44 + len(_CROSS_SECTIONAL_OPERATORS) + len(_TASK33_OPERATORS)
assert len(OPERATOR_REGISTRY.operator_specs) >= _PREV_COUNT + len(_TASK34_OPERATORS), (
    f"OPERATOR_REGISTRY 總數應含既有 {_PREV_COUNT} + Task3.4 {len(_TASK34_OPERATORS)} 個運算元，"
    f"實際 {len(OPERATOR_REGISTRY.operator_specs)}"
)