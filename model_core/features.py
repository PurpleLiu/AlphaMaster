"""
model_core/features.py -- MT5 Feature Engineering (20 features)

Features:
  Trend (0-4):   RET, RET5, RET20, MA_DIFF, SLOPE20
  Volatility (5-8): ATR, RVOL, HL_RANGE, VOL_REGIME
  Reversal (9-13):  DEV, DEV60, RSI14, PRESSURE, AC1
  Volume (14-16):   VOL_RATIO, VOL_Z, PV_CORR
  Cross-asset (17-19): REL_RET5, REL_RET20, REL_VOL

Output: [N, 30, T], all normalized, no NaN/Inf. (v3.0: 20→30 features)

註冊化重構（task 5.1）：現有 30 個特徵以 `FeatureSpec(name, category, compute)`
聲明條目註冊進模組級 `FEATURE_REGISTRY`；`compute_features()` 按註冊順序執行
每個特徵的 compute 並堆疊為 [N, F, T]。計算邏輯與順序與重構前逐元素一致。
每個 compute 的簽名為 `(raw_dict: dict) -> Tensor[N, T]`。
"""
import torch

from .registry import FeatureSpec, Registry


class MT5FeatureEngineer:
    """MT5 Feature Engineer (30 features, v3.0)."""

    INPUT_DIM    = 30  # v3.0: 20→30
    _CLIP_BOUND  = 5.0
    _EPS         = 1e-9
    _MA_WINDOW   = 20
    _NORM_WINDOW = 200  # 因果滾動 robust 歸一化的默認窗口長度（可調）

    # ── rolling helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rolling_mean(x: torch.Tensor, w: int) -> torch.Tensor:
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=1).unfold(1, w, 1).mean(dim=-1)

    @classmethod
    def _ma(cls, x: torch.Tensor, w: int) -> torch.Tensor:
        return cls._rolling_mean(x, w)

    @classmethod
    def _ma20(cls, x: torch.Tensor) -> torch.Tensor:
        return cls._rolling_mean(x, cls._MA_WINDOW)

    @staticmethod
    def _rolling_std(x: torch.Tensor, w: int) -> torch.Tensor:
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        wnd  = torch.cat([pad, x], dim=1).unfold(1, w, 1)
        m    = wnd.mean(dim=-1, keepdim=True)
        return ((wnd - m) ** 2).mean(dim=-1).sqrt() + 1e-9

    @staticmethod
    def _atr(close: torch.Tensor, high: torch.Tensor,
             low: torch.Tensor, w: int = 14) -> torch.Tensor:
        pc = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        tr = torch.stack([high - low,
                          (high - pc).abs(),
                          (low  - pc).abs()], dim=-1).max(dim=-1).values
        return MT5FeatureEngineer._rolling_mean(tr, w)

    @staticmethod
    def _rvol(close: torch.Tensor, w: int = 20) -> torch.Tensor:
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)
        N   = ret.shape[0]
        pad = torch.zeros(N, w - 1, device=ret.device, dtype=ret.dtype)
        wnd = torch.cat([pad, ret], dim=1).unfold(1, w, 1)
        m   = wnd.mean(dim=-1, keepdim=True)
        return ((wnd - m) ** 2).mean(dim=-1).sqrt() + 1e-9

    @staticmethod
    def _ac1(close: torch.Tensor, w: int = 20) -> torch.Tensor:
        eps = MT5FeatureEngineer._EPS
        ret = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret = torch.cat([torch.zeros_like(close[:, :1]), ret], dim=1)
        N   = ret.shape[0]
        pad = torch.zeros(N, w, device=ret.device, dtype=ret.dtype)
        wnd = torch.cat([pad, ret], dim=1).unfold(1, w + 1, 1)
        x, y = wnd[:, :, :-1], wnd[:, :, 1:]
        xm, ym = x.mean(dim=-1, keepdim=True), y.mean(dim=-1, keepdim=True)
        cov = ((x - xm) * (y - ym)).mean(dim=-1)
        sx  = ((x - xm) ** 2).mean(dim=-1).sqrt()
        sy  = ((y - ym) ** 2).mean(dim=-1).sqrt()
        out = cov / (sx * sy + 1e-8)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _linear_slope(x: torch.Tensor, w: int) -> torch.Tensor:
        """Causal linear regression slope, normalized by price level."""
        N, T  = x.shape
        eps   = MT5FeatureEngineer._EPS
        pad   = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        wnd   = torch.cat([pad, x], dim=1).unfold(1, w, 1)   # [N, T, w]
        tidx  = torch.arange(w, dtype=x.dtype, device=x.device)
        tc    = tidx - tidx.mean()
        tvar  = (tc ** 2).sum()
        xm    = wnd.mean(dim=-1, keepdim=True)
        slope = ((wnd - xm) * tc).sum(dim=-1) / (tvar + eps)
        slope = slope / (xm.squeeze(-1) + eps)
        return torch.nan_to_num(slope, nan=0.0)

    @staticmethod
    def _rsi(close: torch.Tensor, w: int = 14) -> torch.Tensor:
        """RSI normalized to [-1, 1]."""
        diff   = close - torch.cat([close[:, :1], close[:, :-1]], dim=1)
        gains  = torch.relu(diff)
        losses = torch.relu(-diff)
        avg_g  = MT5FeatureEngineer._rolling_mean(gains,  w)
        avg_l  = MT5FeatureEngineer._rolling_mean(losses, w)
        rs     = (avg_g + 1e-9) / (avg_l + 1e-9)
        rsi    = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0

    @staticmethod
    def _ts_corr(x: torch.Tensor, y: torch.Tensor, w: int) -> torch.Tensor:
        """Causal sliding Pearson correlation."""
        N, T  = x.shape
        px    = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        py    = torch.zeros(N, w - 1, dtype=y.dtype, device=y.device)
        wx    = torch.cat([px, x], dim=1).unfold(1, w, 1)
        wy    = torch.cat([py, y], dim=1).unfold(1, w, 1)
        mx, my = wx.mean(dim=-1, keepdim=True), wy.mean(dim=-1, keepdim=True)
        cov   = ((wx - mx) * (wy - my)).mean(dim=-1)
        sx    = ((wx - mx) ** 2).mean(dim=-1).sqrt()
        sy    = ((wy - my) ** 2).mean(dim=-1).sqrt()
        mask  = (sx < 1e-6) | (sy < 1e-6)
        corr  = cov / (sx * sy + 1e-8)
        corr[mask] = 0.0
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _robust_norm(x: torch.Tensor, w: int = 200) -> torch.Tensor:
        """因果滾動 robust 歸一化（R1.9, Property 1）。

        每個時間步 t 只使用 [t-w+1..t] 共 w 期數據計算 median/MAD，
        徹底消除 look-ahead 洩露。w 默認 200（覆蓋足夠的歷史，warm-up
        期（t<w）用可用數據的局部 median/MAD，填儲值為 0 不引入未來）。

        【實現注意】：torch.median 對 float16 有精度問題，統一轉 float32 計算
        後再轉回原 dtype。unfold 窗口中有 pad 的 0（warm-up 期），這些 0 會
        影響局部 median，對極短序列略有偏差，但嚴格因果、無未來資訊。
        """
        orig_dtype = x.dtype
        x32 = x.float()
        N, T = x32.shape
        pad = torch.zeros(N, w - 1, device=x32.device, dtype=x32.dtype)
        wnd = torch.cat([pad, x32], dim=1).unfold(1, w, 1)   # [N, T, w]
        med = wnd.median(dim=-1).values                       # [N, T]，因果
        mad = (wnd - med.unsqueeze(-1)).abs().median(dim=-1).values + 1e-6
        out = torch.clamp((x32 - med) / mad,
                          -MT5FeatureEngineer._CLIP_BOUND,
                           MT5FeatureEngineer._CLIP_BOUND)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(orig_dtype)

    @staticmethod
    def _clean(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _ret20(close: torch.Tensor) -> torch.Tensor:
        """相容方法：20期對數動量，前20位補0，返回 [N, T]。

        保留此 helper 以相容外部調用和測試代碼。
        compute_features() 內部等效計算見 ret20_raw。
        """
        N, T = close.shape
        eps  = MT5FeatureEngineer._EPS
        raw  = torch.log(close[:, 20:] / (close[:, :-20] + eps))
        pad  = torch.zeros(N, 20, device=close.device, dtype=close.dtype)
        return torch.cat([pad, raw], dim=1)

    # ── v3.0 新增特徵 helper ───────────────────────────────────────────

    @staticmethod
    def _vwap_dev(close: torch.Tensor, high: torch.Tensor,
                  low: torch.Tensor, volume: torch.Tensor, w: int = 20) -> torch.Tensor:
        """VWAP 偏離: (close - VWAP) / VWAP。VWAP = sum(typical_price * vol) / sum(vol)。"""
        eps = MT5FeatureEngineer._EPS
        typical = (high + low + close) / 3.0
        tpv = typical * volume
        pad_v = torch.zeros(volume.shape[0], w - 1, device=volume.device, dtype=volume.dtype)
        pad_tpv = torch.zeros(tpv.shape[0], w - 1, device=tpv.device, dtype=tpv.dtype)
        vol_w = torch.cat([pad_v, volume], dim=1).unfold(1, w, 1)
        tpv_w = torch.cat([pad_tpv, tpv], dim=1).unfold(1, w, 1)
        vwap = tpv_w.sum(dim=-1) / (vol_w.sum(dim=-1) + eps)
        return (close - vwap) / (vwap + eps)

    @staticmethod
    def _boll_pos(close: torch.Tensor, w: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """布林帶位置[0,1] 和 寬度。返回 (pos, width)。"""
        eps = MT5FeatureEngineer._EPS
        ma = MT5FeatureEngineer._rolling_mean(close, w)
        std = MT5FeatureEngineer._rolling_std(close, w)
        upper = ma + 2 * std
        lower = ma - 2 * std
        pos = (close - lower) / (upper - lower + eps)
        pos = torch.clamp(pos, 0.0, 1.0)
        width = (upper - lower) / (ma + eps)
        return pos, width

    @staticmethod
    def _macd_hist(close: torch.Tensor) -> torch.Tensor:
        """MACD 柱 = (EMA12 - EMA26) - Signal(EMA9 of MACD)。"""
        macd = MT5FeatureEngineer._ema_simple(close, 12) - MT5FeatureEngineer._ema_simple(close, 26)
        signal = MT5FeatureEngineer._ema_simple(macd, 9)
        return macd - signal

    @staticmethod
    def _ema_simple(x: torch.Tensor, span: int, exact: bool = False) -> torch.Tensor:
        """指數加權移動平均（因果），span 期。

        默認路徑（exact=False）：
            向量化因果卷積近似，複雜度 O(N·T·w)，無逐時間步 Python 循環（R8.3）。
            alpha = 2/(span+1)；有效窗口 w = min(T, ceil(-log(1e-6)/(-log(1-alpha))))，
            保證尾部權重 (1-alpha)^w < 1e-6。
            使用首值填充（first-value padding）以匹配遞推版初始條件 out[0]=x[0]，
            max|Δ| 與遞推版差異實測 < 1e-4。

        可選精確路徑（exact=True）：
            嚴格遞推 out[t] = alpha*x[t] + (1-alpha)*out[t-1]。
            複雜度 O(N·T)（順序累積）。
        """
        import math
        alpha = 2.0 / (span + 1.0)
        N, T = x.shape

        if exact:
            out = torch.zeros_like(x)
            out[:, 0] = x[:, 0]
            for t in range(1, T):
                out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
            return out

        # 向量化因果卷積近似（首值填充）
        if alpha >= 1.0:
            return x.clone()
        # w_full 僅由 span 決定，不依賴 T，保證因果性
        w_full = max(1, math.ceil(-math.log(1e-6) / (-math.log(1.0 - alpha))))

        # T < 2*w_full：精確遞推（嚴格因果 O(N·T)）；
        # T >= 2*w_full：向量化卷積近似，首值填充，max|Δ| < 1e-4。
        # 固定閾值 2*w_full 確保不同長度序列超閾值後行為一致。
        if T < 2 * w_full:
            out = torch.zeros_like(x)
            out[:, 0] = x[:, 0]
            for t in range(1, T):
                out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
            return out

        # T >= 2*w_full：向量化，首值填充，max|Δ| < 1e-4
        decay = 1.0 - alpha
        powers = torch.arange(w_full - 1, -1, -1, dtype=x.dtype, device=x.device)
        weights = alpha * (decay ** powers)                    # 未歸一化
        first = x[:, :1].expand(N, w_full - 1)                # [N, w_full-1] 首值填充
        xp = torch.cat([first, x], dim=1)
        windows = xp.unfold(1, w_full, 1)                      # [N, T, w_full]
        out = (windows * weights).sum(dim=-1)                  # [N, T]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _obv_slope(close: torch.Tensor, volume: torch.Tensor, w: int = 20) -> torch.Tensor:
        """能量潮斜率：OBV 的 w 期線性回歸斜率（歸一化）。"""
        eps = MT5FeatureEngineer._EPS
        ret_sign = torch.sign(close[:, 1:] - close[:, :-1])
        ret_sign = torch.cat([torch.zeros_like(close[:, :1]), ret_sign], dim=1)
        obv = torch.cumsum(ret_sign * volume, dim=1)
        # OBV 斜率（用線性回歸）
        return MT5FeatureEngineer._linear_slope(obv, w)

    @staticmethod
    def _mfi(close: torch.Tensor, high: torch.Tensor,
             low: torch.Tensor, volume: torch.Tensor, w: int = 14) -> torch.Tensor:
        """資金流量指標 MFI（帶量版 RSI），歸一化到 [-1, 1]。"""
        eps = MT5FeatureEngineer._EPS
        typical = (high + low + close) / 3.0
        mf = typical * volume  # 資金流量
        pc = torch.cat([typical[:, :1], typical[:, :-1]], dim=1)
        pos_mf = torch.where(typical > pc, mf, torch.zeros_like(mf))
        neg_mf = torch.where(typical < pc, mf, torch.zeros_like(mf))
        pos_sum = MT5FeatureEngineer._rolling_mean(pos_mf, w) * w
        neg_sum = MT5FeatureEngineer._rolling_mean(neg_mf, w) * w
        mfr = pos_sum / (neg_sum + eps)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return (mfi - 50.0) / 50.0

    # ── v3.0 Alpha 101 + 互補特徵 helper ─────────────────────────────

    @staticmethod
    def _willr(close: torch.Tensor, high: torch.Tensor,
               low: torch.Tensor, w: int = 14) -> torch.Tensor:
        """威廉指標 Williams %R，歸一化到 [-1, 0]（-1=超賣，0=超買）。"""
        eps = MT5FeatureEngineer._EPS
        pad = torch.zeros(close.shape[0], w - 1, device=close.device, dtype=high.dtype)
        hw = torch.cat([pad, high], dim=1).unfold(1, w, 1).max(dim=-1).values
        lw = torch.cat([pad, low], dim=1).unfold(1, w, 1).min(dim=-1).values
        willr = (hw - close) / (hw - lw + eps)
        return torch.clamp(willr, -1.0, 0.0)

    @staticmethod
    def _cci(close: torch.Tensor, high: torch.Tensor,
             low: torch.Tensor, w: int = 14) -> torch.Tensor:
        """商品通道指標 CCI = (typical - MA(typical)) / (0.015 * MAD(typical))。"""
        eps = MT5FeatureEngineer._EPS
        typical = (high + low + close) / 3.0
        ma = MT5FeatureEngineer._rolling_mean(typical, w)
        pad = torch.zeros(typical.shape[0], w - 1, device=typical.device, dtype=typical.dtype)
        tw = torch.cat([pad, typical], dim=1).unfold(1, w, 1)
        mad = (tw - tw.mean(dim=-1, keepdim=True)).abs().mean(dim=-1)
        cci = (typical - ma) / (0.015 * mad + eps)
        return torch.clamp(cci / 200.0, -1.0, 1.0)  # 歸一化到 [-1, 1]

    @staticmethod
    def _roc(close: torch.Tensor, w: int = 12) -> torch.Tensor:
        """變化率 ROC = close[t]/close[t-w] - 1，前 w 位補 0。"""
        eps = MT5FeatureEngineer._EPS
        N = close.shape[0]
        raw = close[:, w:] / (close[:, :-w] + eps) - 1.0
        pad = torch.zeros(N, w, device=close.device, dtype=close.dtype)
        return torch.cat([pad, raw], dim=1)

    @staticmethod
    def _typical_dev(close: torch.Tensor, high: torch.Tensor,
                     low: torch.Tensor, w: int = 20) -> torch.Tensor:
        """典型價格 (H+L+C)/3 偏離其 MA_w。與 VWAP_DEV 互補（無成交量加權）。"""
        eps = MT5FeatureEngineer._EPS
        typical = (high + low + close) / 3.0
        ma = MT5FeatureEngineer._rolling_mean(typical, w)
        return (typical - ma) / (ma + eps)

    # ── task 5.2 趨勢/動量類特徵 helper ────────────────────────────────

    @staticmethod
    def _rolling_sum(x: torch.Tensor, w: int) -> torch.Tensor:
        """因果滾動求和（左側 zero-pad，warm-up 用可用數據）。"""
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=1).unfold(1, w, 1).sum(dim=-1)

    @classmethod
    def _trend_strength(cls, x: torch.Tensor, w: int) -> torch.Tensor:
        """SLOPE_w * R²：因果窗口線性回歸斜率（按價位歸一）乘擬合優度 R²∈[0,1]。"""
        N, T  = x.shape
        eps   = cls._EPS
        pad   = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        wnd   = torch.cat([pad, x], dim=1).unfold(1, w, 1)      # [N, T, w]
        tidx  = torch.arange(w, dtype=x.dtype, device=x.device)
        tc    = tidx - tidx.mean()
        tvar  = (tc ** 2).sum()
        xm    = wnd.mean(dim=-1, keepdim=True)                  # [N, T, 1]
        slope = ((wnd - xm) * tc).sum(dim=-1) / (tvar + eps)    # [N, T]
        pred  = xm + slope.unsqueeze(-1) * tc                   # [N, T, w]
        ss_res = ((wnd - pred) ** 2).sum(dim=-1)
        ss_tot = ((wnd - xm) ** 2).sum(dim=-1)
        r2    = torch.clamp(1.0 - ss_res / (ss_tot + eps), 0.0, 1.0)
        slope_norm = slope / (xm.squeeze(-1) + eps)
        return torch.nan_to_num(slope_norm * r2, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _trix(cls, close: torch.Tensor, span: int = 15) -> torch.Tensor:
        """TRIX：三重 EMA 平滑後的單步變化率（因果），首步為 0。"""
        eps = cls._EPS
        e3  = cls._ema_simple(cls._ema_simple(cls._ema_simple(close, span), span), span)
        prev = torch.cat([e3[:, :1], e3[:, :-1]], dim=1)
        return (e3 - prev) / (prev.abs() + eps)

    @classmethod
    def _ppo(cls, close: torch.Tensor) -> torch.Tensor:
        """PPO：百分比價格振盪 (EMA12 - EMA26) / EMA26（因果）。"""
        eps = cls._EPS
        e12 = cls._ema_simple(close, 12)
        e26 = cls._ema_simple(close, 26)
        return (e12 - e26) / (e26.abs() + eps)

    @classmethod
    def _ult_osc(cls, close: torch.Tensor, high: torch.Tensor,
                 low: torch.Tensor) -> torch.Tensor:
        """Ultimate Oscillator：3 週期(7/14/28)加權買壓比，映射到 [-1, 1]。"""
        eps = cls._EPS
        pc  = torch.cat([close[:, :1], close[:, :-1]], dim=1)   # 前收盤（因果）
        true_low  = torch.minimum(low, pc)
        true_high = torch.maximum(high, pc)
        bp = close - true_low                                    # buying pressure
        tr = true_high - true_low                                # true range

        def _avg(w: int) -> torch.Tensor:
            return cls._rolling_sum(bp, w) / (cls._rolling_sum(tr, w) + eps)

        uo = (4.0 * _avg(7) + 2.0 * _avg(14) + _avg(28)) / 7.0   # ∈ [0, 1]
        return torch.clamp(uo * 2.0 - 1.0, -1.0, 1.0)            # → [-1, 1]

    # ── 歸一化封裝（與原 compute_features 內聯 norm 一致）───────────────

    @classmethod
    def _norm(cls, x: torch.Tensor) -> torch.Tensor:
        """因果 robust_norm 歸一化：clip 到 [-5,5]，出入口各 clean 一次。"""
        return cls._clean(cls._robust_norm(cls._clean(x), cls._NORM_WINDOW))

    # ── per-feature compute（註冊條目，簽名 (raw_dict) -> [N, T]）────────
    # 每個 compute 與原 compute_features 的對應內聯片段逐元素等價。共享中間量
    # 的特徵（PV_CORR / REL_*）在各自 compute 內重算其依賴，保持數值一致。

    # 趨勢類 trend (0-4)
    @classmethod
    def _c_ret(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        N = close.shape[0]; eps = cls._EPS
        ret_raw = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        return cls._norm(torch.cat([torch.zeros(N, 1, device=close.device), ret_raw], dim=1))

    @classmethod
    def _c_ret5(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        N = close.shape[0]; eps = cls._EPS
        ret5_raw = torch.log(close[:, 5:] / (close[:, :-5] + eps))
        return cls._norm(torch.cat([torch.zeros(N, 5, device=close.device), ret5_raw], dim=1))

    @classmethod
    def _c_ret20(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        N = close.shape[0]; eps = cls._EPS
        ret20_raw = torch.log(close[:, 20:] / (close[:, :-20] + eps))
        return cls._norm(torch.cat([torch.zeros(N, 20, device=close.device), ret20_raw], dim=1))

    @classmethod
    def _c_ma_diff(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); eps = cls._EPS
        return cls._norm(cls._ma(close, 10) / (cls._ma(close, 30) + eps) - 1.0)

    @classmethod
    def _c_slope20(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._norm(cls._linear_slope(close, 20))

    # 波動類 volatility (5-8)
    @classmethod
    def _c_atr(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        atr_raw = cls._atr(close, high, low)
        return cls._norm(torch.log1p(cls._clean(atr_raw.clamp(min=0))))

    @classmethod
    def _c_rvol(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        rvol_raw = cls._rvol(close)
        return cls._norm(torch.log1p(cls._clean(rvol_raw.clamp(min=0))))

    @classmethod
    def _c_hl_range(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        eps = cls._EPS
        return cls._norm((high - low) / (close + eps))

    @classmethod
    def _c_vol_regime(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        eps = cls._EPS
        atr_raw = cls._atr(close, high, low)
        ma_atr = cls._ma(atr_raw, 20)
        return cls._norm(atr_raw / (ma_atr + eps) - 1.0)

    # 反轉類 reversal (9-13)
    @classmethod
    def _c_dev(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); eps = cls._EPS
        ma20c = cls._ma20(close)
        return cls._norm((close - ma20c) / (ma20c + eps))

    @classmethod
    def _c_dev60(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); eps = cls._EPS
        ma60 = cls._ma(close, 60)
        return cls._norm((close - ma60) / (ma60 + eps))

    @classmethod
    def _c_rsi14(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._clean(torch.clamp(cls._rsi(close, 14), -1.0, 1.0))

    @classmethod
    def _c_pressure(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); open_ = raw["open"].float()
        high = raw["high"].float(); low = raw["low"].float()
        eps = cls._EPS
        return cls._clean(torch.clamp((close - open_) / (high - low + eps), -1.0, 1.0))

    @classmethod
    def _c_ac1(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._clean(torch.clamp(cls._ac1(close), -1.0, 1.0))

    # 成交量類 volume (14-16)
    @classmethod
    def _c_vol_ratio(cls, raw: dict) -> torch.Tensor:
        volume = raw["volume"].float(); eps = cls._EPS
        ma20v = cls._ma20(volume)
        return cls._norm(volume / (ma20v + eps))

    @classmethod
    def _c_vol_z(cls, raw: dict) -> torch.Tensor:
        volume = raw["volume"].float(); eps = cls._EPS
        ma20v = cls._ma20(volume)
        std20v = cls._rolling_std(volume, 20)
        return cls._clean(torch.clamp((volume - ma20v) / (std20v + eps), -5.0, 5.0))

    @classmethod
    def _c_pv_corr(cls, raw: dict) -> torch.Tensor:
        # 依賴歸一化後的 RET（feature 0）與 VOL_RATIO（feature 14），逐元素重算
        ret = cls._c_ret(raw)
        vol_ratio = cls._c_vol_ratio(raw)
        log_vol_ratio = torch.log1p(cls._clean(vol_ratio.clamp(min=-0.99)))
        return cls._clean(torch.clamp(cls._ts_corr(ret, log_vol_ratio, 10), -1.0, 1.0))

    # 跨截面相對強弱 cross_sectional (17-19)
    @classmethod
    def _c_rel_ret5(cls, raw: dict) -> torch.Tensor:
        ret5 = cls._c_ret5(raw)
        return cls._norm(ret5 - ret5.mean(dim=0, keepdim=True))

    @classmethod
    def _c_rel_ret20(cls, raw: dict) -> torch.Tensor:
        ret20 = cls._c_ret20(raw)
        return cls._norm(ret20 - ret20.mean(dim=0, keepdim=True))

    @classmethod
    def _c_rel_vol(cls, raw: dict) -> torch.Tensor:
        rvol = cls._c_rvol(raw)
        return cls._norm(rvol - rvol.mean(dim=0, keepdim=True))

    # v3.0 新增特徵 (20-25)
    @classmethod
    def _c_vwap_dev(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float()
        low = raw["low"].float(); volume = raw["volume"].float()
        return cls._norm(cls._vwap_dev(close, high, low, volume))

    @classmethod
    def _c_boll_pos(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        boll_pos, _ = cls._boll_pos(close)
        return cls._clean(boll_pos)

    @classmethod
    def _c_boll_width(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        _, boll_width = cls._boll_pos(close)
        return cls._norm(boll_width)

    @classmethod
    def _c_macd_hist(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._norm(cls._macd_hist(close))

    @classmethod
    def _c_obv_slope(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); volume = raw["volume"].float()
        return cls._norm(cls._obv_slope(close, volume))

    @classmethod
    def _c_mfi14(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float()
        low = raw["low"].float(); volume = raw["volume"].float()
        return cls._clean(torch.clamp(cls._mfi(close, high, low, volume), -1.0, 1.0))

    # v3.0 Alpha 101 + 互補特徵 (26-29)
    @classmethod
    def _c_willr14(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        return cls._clean(cls._willr(close, high, low))

    @classmethod
    def _c_cci14(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        return cls._clean(cls._cci(close, high, low))

    @classmethod
    def _c_roc12(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._norm(cls._roc(close, 12))

    @classmethod
    def _c_typical_dev(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        return cls._norm(cls._typical_dev(close, high, low))

    # ── task 5.2 趨勢類 trend (30-32) ─────────────────────────────────
    @classmethod
    def _c_ema_ratio_12_26(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); eps = cls._EPS
        e12 = cls._ema_simple(close, 12)
        e26 = cls._ema_simple(close, 26)
        return cls._norm(e12 / (e26 + eps) - 1.0)

    @classmethod
    def _c_trend_strength_50(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._norm(cls._trend_strength(close, 50))

    @classmethod
    def _c_price_pos_50(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        eps = cls._EPS; w = 50
        pad = torch.zeros(close.shape[0], w - 1, device=close.device, dtype=close.dtype)
        max50 = torch.cat([pad, high], dim=1).unfold(1, w, 1).max(dim=-1).values
        min50 = torch.cat([pad, low],  dim=1).unfold(1, w, 1).min(dim=-1).values
        pos = (close - min50) / (max50 - min50 + eps)
        return cls._clean(torch.clamp(pos, 0.0, 1.0))

    # ── task 5.3 波動類（含 OHLC 估計量）features (37-40) ─────────────────
    # 所有滾動均值用 _rolling_mean（因果左pad+unfold），嚴格無未來資訊（R1.9）。
    # 所有 log 的分母加 eps 防零，clamp 防負數 log；warm-up 期填 0（nan_to_num）。
    # 歸一化：先 log1p(raw.clamp(min=0))（對非負原始波動值），再 _norm()（R1.10）。

    @classmethod
    def _c_gk_vol(cls, raw: dict) -> torch.Tensor:
        """Garman-Klass 波動率估計量（R1.3）。

        逐 bar 計算：GK = 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
        然後滾動均值（w=20）取 sqrt。因果實現，無未來資訊。
        """
        eps   = cls._EPS
        open_ = raw["open"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        close = raw["close"].float()

        ln2     = torch.log(torch.tensor(2.0, device=close.device))
        hl_term = 0.5 * (torch.log((high + eps) / (low + eps))) ** 2
        co_term = (2.0 * ln2 - 1.0) * (torch.log((close + eps) / (open_ + eps))) ** 2
        gk_bar  = (hl_term - co_term).clamp(min=0.0)          # 每 bar 的 GK 估計量（≥0）

        gk_mean = cls._rolling_mean(gk_bar, 20)               # 因果滾動均值（w=20）
        raw_vol = gk_mean.sqrt()                               # 取 sqrt → 波動率尺度
        return cls._norm(torch.log1p(cls._clean(raw_vol)))

    @classmethod
    def _c_parkinson_vol(cls, raw: dict) -> torch.Tensor:
        """Parkinson 波動率估計量（R1.3）。

        逐 bar 計算：PK = (1/(4*ln2)) * (ln(H/L))^2
        然後滾動均值（w=20）取 sqrt。
        """
        eps  = cls._EPS
        high = raw["high"].float()
        low  = raw["low"].float()

        ln2    = torch.log(torch.tensor(2.0, device=high.device))
        pk_bar = (1.0 / (4.0 * ln2)) * (torch.log((high + eps) / (low + eps))) ** 2
        pk_bar = pk_bar.clamp(min=0.0)

        pk_mean = cls._rolling_mean(pk_bar, 20)
        raw_vol = pk_mean.sqrt()
        return cls._norm(torch.log1p(cls._clean(raw_vol)))

    @classmethod
    def _c_yang_zhang_vol(cls, raw: dict) -> torch.Tensor:
        """Yang-Zhang 波動率估計量（因果版，R1.3）。

        YZ = overnight_vol + k*open_vol + (1-k)*RS_vol，等權簡化實現。
        其中:
          overnight_vol：ln(O[t]/C[t-1])^2 的滾動均值
          open_vol：ln(O[t]/C[t])^2 的滾動均值
          RS_vol：Rogers-Satchell = ln(H/C)*ln(H/O)+ln(L/C)*ln(L/O) 的滾動均值
        前收盤 pc = cat([close[:,:1], close[:,:-1]])（因果，無洩露）。
        簡化為等權平均：YZ = (overnight + open + RS) / 3。
        """
        eps   = cls._EPS
        open_ = raw["open"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        close = raw["close"].float()

        # 前收盤（因果）：第一步用當前收盤填充（warm-up 期合理退化）
        pc = torch.cat([close[:, :1], close[:, :-1]], dim=1)   # [N, T]，因果

        # overnight: ln(O/C_prev)^2
        overnight_bar = (torch.log((open_ + eps) / (pc + eps))) ** 2

        # open vol: ln(O/C)^2（當日開盤相對收盤）
        open_bar = (torch.log((open_ + eps) / (close + eps))) ** 2

        # Rogers-Satchell 逐 bar
        rs_bar = (
            torch.log((high + eps) / (close + eps)) * torch.log((high + eps) / (open_ + eps))
            + torch.log((low  + eps) / (close + eps)) * torch.log((low  + eps) / (open_ + eps))
        ).clamp(min=0.0)

        # 等權平均後滾動均值，clamp(min=0) 取 sqrt
        yz_bar  = (overnight_bar + open_bar + rs_bar) / 3.0
        yz_mean = cls._rolling_mean(yz_bar, 20)
        raw_vol = yz_mean.clamp(min=0.0).sqrt()
        return cls._norm(torch.log1p(cls._clean(raw_vol)))

    @classmethod
    def _c_rs_vol(cls, raw: dict) -> torch.Tensor:
        """Rogers-Satchell 波動率估計量（R1.3）。

        逐 bar 計算：RS = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
        滾動均值（w=20）clamp(min=0) 後取 sqrt（RS 可能為負，clamp 保安全）。
        """
        eps   = cls._EPS
        open_ = raw["open"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        close = raw["close"].float()

        rs_bar = (
            torch.log((high + eps) / (close + eps)) * torch.log((high + eps) / (open_ + eps))
            + torch.log((low  + eps) / (close + eps)) * torch.log((low  + eps) / (open_ + eps))
        )
        rs_mean = cls._rolling_mean(rs_bar, 20)
        raw_vol = rs_mean.clamp(min=0.0).sqrt()
        return cls._norm(torch.log1p(cls._clean(raw_vol)))

    # ── task 5.2 動量類 momentum (33-36) ──────────────────────────────
    @classmethod
    def _c_trix_15(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._norm(cls._trix(close, 15))

    @classmethod
    def _c_ppo(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float()
        return cls._norm(cls._ppo(close))

    @classmethod
    def _c_ult_osc(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); high = raw["high"].float(); low = raw["low"].float()
        return cls._clean(cls._ult_osc(close, high, low))

    @classmethod
    def _c_ret_accel(cls, raw: dict) -> torch.Tensor:
        close = raw["close"].float(); eps = cls._EPS; N = close.shape[0]
        ret5_raw = torch.log(close[:, 5:] / (close[:, :-5] + eps))
        ret5 = torch.cat([torch.zeros(N, 5, device=close.device, dtype=close.dtype), ret5_raw], dim=1)
        pad = torch.zeros(N, 5, device=close.device, dtype=close.dtype)
        prev = torch.cat([pad, ret5[:, :-5]], dim=1)             # ret5[t-5]，前5位為0
        return cls._norm(ret5 - prev)

    # ── task 5.4 量能/流動性類特徵 volume (41-44) ─────────────────────────
    # 所有滾動操作用 _rolling_mean/_rolling_sum/_linear_slope（因果左pad+unfold）。
    # cumsum 是嚴格因果的。所有除法/log 加 eps（R8.1）；出口 nan_to_num（R8.6）。

    @classmethod
    def _c_amihud_illiq(cls, raw: dict) -> torch.Tensor:
        """Amihud 非流動性指標（R1.6）：mean(|ret|/(volume+eps))，滾動 w=20。

        逐 bar：illiq[t] = |log_ret[t]| / (volume[t] + eps)
        因果滾動均值（w=20），log1p→robust_norm。
        """
        eps    = cls._EPS
        close  = raw["close"].float()
        volume = raw["volume"].float()
        N      = close.shape[0]

        log_ret_raw = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        log_ret = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), log_ret_raw], dim=1)
        illiq_bar = log_ret.abs() / (volume + eps)                # [N, T]
        illiq_mean = cls._rolling_mean(illiq_bar, 20)             # 因果滾動均值
        return cls._norm(torch.log1p(cls._clean(illiq_mean.clamp(min=0))))

    @classmethod
    def _c_kyle_lambda(cls, raw: dict) -> torch.Tensor:
        """Kyle lambda 近似（價格衝擊斜率，R1.6）。

        在過去 w=20 期窗口內：
          signed_vol = ret.sign() * volume
          lambda = cov(|ret|, signed_vol) / (var(signed_vol) + eps)
        用因果 unfold 實現滾動協方差/方差。
        """
        eps    = cls._EPS
        close  = raw["close"].float()
        volume = raw["volume"].float()
        N      = close.shape[0]
        w      = 20

        log_ret_raw = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        log_ret = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), log_ret_raw], dim=1)
        abs_ret     = log_ret.abs()
        signed_vol  = log_ret.sign() * volume

        pad_a  = torch.zeros(N, w - 1, device=close.device, dtype=close.dtype)
        pad_s  = torch.zeros(N, w - 1, device=close.device, dtype=close.dtype)
        wa = torch.cat([pad_a, abs_ret],    dim=1).unfold(1, w, 1)   # [N, T, w]
        ws = torch.cat([pad_s, signed_vol], dim=1).unfold(1, w, 1)   # [N, T, w]

        ma = wa.mean(dim=-1, keepdim=True)
        ms = ws.mean(dim=-1, keepdim=True)
        cov = ((wa - ma) * (ws - ms)).mean(dim=-1)                    # [N, T]
        var_s = ((ws - ms) ** 2).mean(dim=-1)                         # [N, T]
        lam = cov / (var_s + eps)
        return cls._norm(cls._clean(lam))

    @classmethod
    def _c_cmf_20(cls, raw: dict) -> torch.Tensor:
        """Chaikin Money Flow 20 期（R1.6）。

        逐 bar：MFV = ((C-L)-(H-C))/(H-L+eps) * volume
        CMF = sum(MFV, 20) / (sum(volume, 20) + eps)，值域 [-1, 1]。
        因果 _rolling_sum（左pad+unfold）。
        """
        eps    = cls._EPS
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        volume = raw["volume"].float()

        mf_mul = ((close - low) - (high - close)) / (high - low + eps)  # ∈ [-1, 1]
        mfv    = mf_mul * volume                                          # [N, T]
        cmf    = cls._rolling_sum(mfv, 20) / (cls._rolling_sum(volume, 20) + eps)
        return cls._clean(torch.clamp(cmf, -1.0, 1.0))

    @classmethod
    def _c_ad_line_slope(cls, raw: dict) -> torch.Tensor:
        """A/D line 斜率（R1.6）。

        A/D line = cumsum(((C-L)-(H-C))/(H-L+eps) * volume)（嚴格因果）。
        取其滾動線性回歸斜率（_linear_slope 因果 unfold），robust_norm。
        """
        eps    = cls._EPS
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        volume = raw["volume"].float()

        mf_mul  = ((close - low) - (high - close)) / (high - low + eps)
        mfv     = mf_mul * volume
        ad_line = torch.cumsum(mfv, dim=1)                           # 嚴格因果
        slope   = cls._linear_slope(ad_line, 20)
        return cls._norm(cls._clean(slope))

    # ── task 5.5 反轉/振盪類特徵 reversal + trend/momentum (45-50) ────────
    # 所有滾動極值/均值用因果 unfold；prev_H/prev_L 用 cat(...,:-1) 獲取（因果）。

    @classmethod
    def _c_stoch_k_14(cls, raw: dict) -> torch.Tensor:
        """Stochastic %K（14 期，R1.2 reversal）。

        %K = (C - min(L, 14)) / (max(H, 14) - min(L, 14) + eps)
        分子/分母均用因果 unfold（14 期滾動極值）。
        線性映射到 [-1, 1]：%K * 2 - 1。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        N     = close.shape[0]
        w     = 14

        pad   = torch.zeros(N, w - 1, device=close.device, dtype=close.dtype)
        max_h = torch.cat([pad, high], dim=1).unfold(1, w, 1).max(dim=-1).values   # [N, T]
        min_l = torch.cat([pad, low],  dim=1).unfold(1, w, 1).min(dim=-1).values   # [N, T]
        pct_k = (close - min_l) / (max_h - min_l + eps)
        out   = torch.clamp(pct_k * 2.0 - 1.0, -1.0, 1.0)
        return cls._clean(out)

    @classmethod
    def _c_stoch_d_3(cls, raw: dict) -> torch.Tensor:
        """%K 的 3 期均值信號線（R1.2 reversal）。

        %D = MA3(%K)，_rolling_mean 因果實現，映射到 [-1, 1]。
        """
        # 先算 %K（未 clamp 前的原始值 ∈ [0,1]）
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        N     = close.shape[0]
        w     = 14

        pad   = torch.zeros(N, w - 1, device=close.device, dtype=close.dtype)
        max_h = torch.cat([pad, high], dim=1).unfold(1, w, 1).max(dim=-1).values
        min_l = torch.cat([pad, low],  dim=1).unfold(1, w, 1).min(dim=-1).values
        pct_k = (close - min_l) / (max_h - min_l + eps)              # [0, 1]
        pct_d = cls._rolling_mean(pct_k, 3)                           # 因果 MA3
        out   = torch.clamp(pct_d * 2.0 - 1.0, -1.0, 1.0)
        return cls._clean(out)

    @classmethod
    def _c_aroon_osc_25(cls, raw: dict) -> torch.Tensor:
        """Aroon 振盪器 25 期（R1.2 reversal）。

        Aroon_Up   = (25 - periods_since_high_25) / 25 * 100
        Aroon_Down = (25 - periods_since_low_25)  / 25 * 100
        Aroon_Osc  = (Aroon_Up - Aroon_Down) / 100，值域 [-1, 1]。
        用 argmax/argmin 在因果 25 期窗口內找最高/最低價位置。
        argmax 返回 [0..w-1]（0=窗口最早，w-1=最近），
        periods_since = (w-1) - argmax。
        """
        high  = raw["high"].float()
        low   = raw["low"].float()
        N     = high.shape[0]
        w     = 25

        pad_h = torch.zeros(N, w - 1, device=high.device, dtype=high.dtype)
        pad_l = torch.zeros(N, w - 1, device=low.device,  dtype=low.dtype)
        wh    = torch.cat([pad_h, high], dim=1).unfold(1, w, 1)   # [N, T, w]
        wl    = torch.cat([pad_l, low],  dim=1).unfold(1, w, 1)

        # argmax 返回最近位置的索引（0=最早，w-1=最近）
        idx_h = wh.argmax(dim=-1).float()                          # [N, T]
        idx_l = wl.argmin(dim=-1).float()

        # periods_since_high = (w-1) - idx_h（0 表示當前 bar 是最高，w-1 表示很久前）
        periods_h = (w - 1) - idx_h
        periods_l = (w - 1) - idx_l

        aroon_up   = (w - periods_h) / w                          # ∈ [0, 1]
        aroon_down = (w - periods_l) / w
        osc        = aroon_up - aroon_down                         # ∈ [-1, 1]
        return cls._clean(torch.clamp(osc, -1.0, 1.0))

    @classmethod
    def _c_dmi_adx_14(cls, raw: dict) -> torch.Tensor:
        """ADX 趨勢強度（14 期，R1.2 trend）。

        DM+ = max(H-prev_H, 0)，DM- = max(prev_L-L, 0)，TR = ATR 每 bar 分量。
        DI+ = rolling_mean(DM+, 14) / rolling_mean(TR, 14) + eps
        DI- = rolling_mean(DM-, 14) / rolling_mean(TR, 14) + eps
        DX  = |DI+ - DI-| / (DI+ + DI- + eps)
        ADX = rolling_mean(DX, 14)，值域 [0, 1]。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        w     = 14

        # prev_H / prev_L（因果）：第一步用當前 bar 填充
        prev_h = torch.cat([high[:, :1], high[:, :-1]], dim=1)
        prev_l = torch.cat([low[:, :1],  low[:, :-1]],  dim=1)
        prev_c = torch.cat([close[:, :1], close[:, :-1]], dim=1)

        dm_pos = torch.clamp(high - prev_h, min=0.0)
        dm_neg = torch.clamp(prev_l - low,  min=0.0)

        tr = torch.stack([
            high - low,
            (high - prev_c).abs(),
            (low  - prev_c).abs()
        ], dim=-1).max(dim=-1).values                             # [N, T]

        tr_mean  = cls._rolling_mean(tr,     w)                  # [N, T]
        di_pos   = cls._rolling_mean(dm_pos, w) / (tr_mean + eps)
        di_neg   = cls._rolling_mean(dm_neg, w) / (tr_mean + eps)

        dx       = (di_pos - di_neg).abs() / (di_pos + di_neg + eps)
        adx      = cls._rolling_mean(dx, w)
        return cls._clean(torch.clamp(adx, 0.0, 1.0))

    @classmethod
    def _c_dmi_diff_14(cls, raw: dict) -> torch.Tensor:
        """(DI+) - (DI-) 歸一化，值域 [-1, 1]（R1.2 trend）。"""
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        w     = 14

        prev_h = torch.cat([high[:, :1], high[:, :-1]], dim=1)
        prev_l = torch.cat([low[:, :1],  low[:, :-1]],  dim=1)
        prev_c = torch.cat([close[:, :1], close[:, :-1]], dim=1)

        dm_pos = torch.clamp(high - prev_h, min=0.0)
        dm_neg = torch.clamp(prev_l - low,  min=0.0)
        tr = torch.stack([
            high - low,
            (high - prev_c).abs(),
            (low  - prev_c).abs()
        ], dim=-1).max(dim=-1).values

        tr_mean = cls._rolling_mean(tr,     w)
        di_pos  = cls._rolling_mean(dm_pos, w) / (tr_mean + eps)
        di_neg  = cls._rolling_mean(dm_neg, w) / (tr_mean + eps)
        diff    = di_pos - di_neg
        return cls._clean(torch.clamp(diff, -1.0, 1.0))

    @classmethod
    def _c_trix_signal(cls, raw: dict) -> torch.Tensor:
        """TRIX 與其 9 期信號線之差（R1.2 momentum）。

        TRIX  = _trix(close, 15)（已實現，三重EMA變化率）
        signal = _rolling_mean(trix, 9)（因果）
        TRIX_SIGNAL = trix - signal，robust_norm。
        """
        close  = raw["close"].float()
        trix   = cls._trix(close, 15)
        signal = cls._rolling_mean(trix, 9)
        return cls._norm(cls._clean(trix - signal))

    # ── task 5.6 通道/突破類特徵 helpers & compute ────────────────────────

    @staticmethod
    def _rolling_max(x: torch.Tensor, w: int) -> torch.Tensor:
        """因果滾動最大值（左側 zero-pad）。"""
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=1).unfold(1, w, 1).max(dim=-1).values

    @staticmethod
    def _rolling_min(x: torch.Tensor, w: int) -> torch.Tensor:
        """因果滾動最小值（左側 zero-pad）。"""
        N, T = x.shape
        pad  = torch.zeros(N, w - 1, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=1).unfold(1, w, 1).min(dim=-1).values

    @classmethod
    def _c_donchian_pos_20(cls, raw: dict) -> torch.Tensor:
        """Donchian 通道位置：(close - min20) / (max20 - min20 + eps)，clamp[0,1]。

        max20 = rolling_max(high, 20)，min20 = rolling_min(low, 20)（因果 unfold）。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        max20 = cls._rolling_max(high, 20)
        min20 = cls._rolling_min(low,  20)
        pos   = (close - min20) / (max20 - min20 + eps)
        return cls._clean(torch.clamp(pos, 0.0, 1.0))

    @classmethod
    def _c_keltner_pos_20(cls, raw: dict) -> torch.Tensor:
        """Keltner 通道位置：(close - lower) / (upper - lower + eps)，clamp[0,1]。

        mid = EMA(close, 20)，range = EMA(ATR14, 20)（因果）。
        upper = mid + 2*range，lower = mid - 2*range。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        mid   = cls._ema_simple(close, 20)
        atr   = cls._atr(close, high, low, w=14)
        rng   = cls._ema_simple(atr, 20)
        upper = mid + 2.0 * rng
        lower = mid - 2.0 * rng
        pos   = (close - lower) / (upper - lower + eps)
        return cls._clean(torch.clamp(pos, 0.0, 1.0))

    @classmethod
    def _c_ichimoku_kijun_dev(cls, raw: dict) -> torch.Tensor:
        """close 相對 Kijun-sen（26期高低價中值）偏離，robust_norm。

        kijun = (rolling_max(high,26) + rolling_min(low,26)) / 2（因果 unfold）。
        dev = (close - kijun) / (kijun + eps)。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        kijun = (cls._rolling_max(high, 26) + cls._rolling_min(low, 26)) / 2.0
        dev   = (close - kijun) / (kijun + eps)
        return cls._norm(cls._clean(dev))

    @classmethod
    def _c_ichimoku_tenkan_dev(cls, raw: dict) -> torch.Tensor:
        """close 相對 Tenkan-sen（9期高低價中值）偏離，robust_norm。

        tenkan = (rolling_max(high,9) + rolling_min(low,9)) / 2（因果 unfold）。
        dev = (close - tenkan) / (tenkan + eps)。
        """
        eps    = cls._EPS
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        tenkan = (cls._rolling_max(high, 9) + cls._rolling_min(low, 9)) / 2.0
        dev    = (close - tenkan) / (tenkan + eps)
        return cls._norm(cls._clean(dev))

    @classmethod
    def _c_supertrend_dir(cls, raw: dict) -> torch.Tensor:
        """SuperTrend 方向標誌 {-1.0, +1.0}（因果逐時間步遞推）。

        upper_band = (high+low)/2 + 1.5*ATR14（當前 bar）
        lower_band = (high+low)/2 - 1.5*ATR14
        遞推規則（嚴格因果，對每個時間步平行處理所有 N 品種）：
          - t=0：direction = +1
          - t≥1：close > prev_upper_band → +1；close < prev_lower_band → -1；否則保持
        註：Python 循環僅遍歷時間維 T，每步操作 N 品種的向量（合法向量化）。
        """
        close = raw["close"].float()
        high  = raw["high"].float()
        low   = raw["low"].float()
        N, T  = close.shape

        atr          = cls._atr(close, high, low, w=14)
        mid          = (high + low) / 2.0
        upper_band   = mid + 1.5 * atr    # [N, T]
        lower_band   = mid - 1.5 * atr    # [N, T]

        # 遞推 direction：+1=上漲趨勢，-1=下跌趨勢
        direction = torch.ones(N, T, dtype=close.dtype, device=close.device)
        prev_upper = upper_band[:, 0]     # [N]（初始化為 t=0 的帶值）
        prev_lower = lower_band[:, 0]
        for t in range(1, T):
            # 嚴格因果：僅使用 close[t] 與前一時間步的帶值
            flip_up   = close[:, t] > prev_upper   # 價格突破上帶 → 上漲
            flip_down = close[:, t] < prev_lower   # 價格跌破下帶 → 下跌
            prev_dir  = direction[:, t - 1]
            new_dir   = prev_dir.clone()
            new_dir[flip_up]   =  1.0
            new_dir[flip_down] = -1.0
            direction[:, t]    = new_dir
            prev_upper = upper_band[:, t]
            prev_lower = lower_band[:, t]
        return cls._clean(direction)

    @classmethod
    def _c_sar_dist(cls, raw: dict) -> torch.Tensor:
        """close 相對拋物線 SAR 的歸一化距離（簡化實現，嚴格因果）。

        簡化 SAR ≈ EMA(close,20) - EMA(close,50)（方向資訊），robust_norm。
        用快慢 EMA 差作為 SAR 位置的近似——快線 > 慢線 → 上漲；快線 < 慢線 → 下跌。
        """
        close    = raw["close"].float()
        ema20    = cls._ema_simple(close, 20)
        ema50    = cls._ema_simple(close, 50)
        sar_approx = ema20 - ema50
        return cls._norm(cls._clean(sar_approx))

    # ── task 5.7 統計類特徵 helpers & compute ─────────────────────────────

    @classmethod
    def _c_roll_skew_20(cls, raw: dict) -> torch.Tensor:
        """20 期收益率偏度（三階標準矩），robust_norm。

        ret = log(close[t]/close[t-1]+eps)，前1位補0。
        unfold(w=20) 計算三階矩/std^3，因果。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret   = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret   = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), ret], dim=1)
        w     = 20
        pad   = torch.zeros(N, w - 1, device=ret.device, dtype=ret.dtype)
        wnd   = torch.cat([pad, ret], dim=1).unfold(1, w, 1)   # [N, T, w]
        mean  = wnd.mean(dim=-1, keepdim=True)
        diff  = wnd - mean
        std   = diff.pow(2).mean(dim=-1).sqrt()                 # [N, T]
        skew  = diff.pow(3).mean(dim=-1) / (std.pow(3) + eps)  # [N, T]
        return cls._norm(cls._clean(skew))

    @classmethod
    def _c_roll_kurt_20(cls, raw: dict) -> torch.Tensor:
        """20 期收益率峰度（excess kurtosis = 四階矩/std^4 - 3），robust_norm。

        同 _c_roll_skew_20，改為四階矩 / std^4 - 3（超額峰度）。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret   = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret   = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), ret], dim=1)
        w     = 20
        pad   = torch.zeros(N, w - 1, device=ret.device, dtype=ret.dtype)
        wnd   = torch.cat([pad, ret], dim=1).unfold(1, w, 1)   # [N, T, w]
        mean  = wnd.mean(dim=-1, keepdim=True)
        diff  = wnd - mean
        std   = diff.pow(2).mean(dim=-1).sqrt()                 # [N, T]
        kurt  = diff.pow(4).mean(dim=-1) / (std.pow(4) + eps) - 3.0
        return cls._norm(cls._clean(kurt))

    @classmethod
    def _c_hurst_50(cls, raw: dict) -> torch.Tensor:
        """50 期 Hurst 指數（R/S 法，因果 unfold，全程向量化），映射到 [-1,1]。

        在過去50期收益窗口內：
          累積偏差序列 = cumsum(x - mean(x))（沿窗口 dim=-1）
          R = max(cumsum) - min(cumsum)（範圍）
          S = std(x)（標準差）
          Hurst ≈ log(R/S) / log(50)（簡化 Hurst）
        映射：2*H - 1 ∈ [-1, 1]（H∈[0,1]）。
        """
        import math
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret   = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret   = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), ret], dim=1)
        w     = 50
        pad   = torch.zeros(N, w - 1, device=ret.device, dtype=ret.dtype)
        wnd   = torch.cat([pad, ret], dim=1).unfold(1, w, 1)   # [N, T, w]
        # 每窗口內中心化序列（沿 dim=-1 即窗口維）
        mean  = wnd.mean(dim=-1, keepdim=True)                  # [N, T, 1]
        centered = wnd - mean                                   # [N, T, w]
        # R/S 分析：cumsum 沿窗口維
        cumdev = torch.cumsum(centered, dim=-1)                 # [N, T, w]
        R = cumdev.max(dim=-1).values - cumdev.min(dim=-1).values   # [N, T]
        S = centered.pow(2).mean(dim=-1).sqrt()                     # std
        hurst = torch.log(R / (S + eps) + eps) / math.log(w)       # ≈ Hurst ∈ [0,1]
        hurst = torch.clamp(hurst, 0.0, 1.0)
        return cls._clean(hurst * 2.0 - 1.0)                   # 映射到 [-1, 1]

    @classmethod
    def _c_fractal_dim_30(cls, raw: dict) -> torch.Tensor:
        """分形維（約≈2-Hurst 的近似，30期窗口），映射到 [-1,1]。

        在過去30期 close 窗口內：
          frac = (max - min) / (mean_abs_diff * sqrt(30) + eps)
          其中 mean_abs_diff = mean(|close[t]-close[t-1]|, 30)（因果 unfold）
        線性映射到 [-1, 1]：clamp [0, 3]，(frac/3)*2-1。
        """
        import math
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        w     = 30
        # close 窗口
        pad_c = torch.zeros(N, w - 1, device=close.device, dtype=close.dtype)
        wnd_c = torch.cat([pad_c, close], dim=1).unfold(1, w, 1)   # [N, T, w]
        rng   = wnd_c.max(dim=-1).values - wnd_c.min(dim=-1).values  # [N, T]
        # 逐差絕對值窗口（相鄰差 = close[t]-close[t-1]）
        diff  = (close[:, 1:] - close[:, :-1]).abs()
        diff  = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), diff], dim=1)
        pad_d = torch.zeros(N, w - 1, device=diff.device, dtype=diff.dtype)
        wnd_d = torch.cat([pad_d, diff], dim=1).unfold(1, w, 1)   # [N, T, w]
        mad   = wnd_d.mean(dim=-1)                                  # [N, T]
        frac  = rng / (mad * math.sqrt(w) + eps)
        frac  = torch.clamp(frac, 0.0, 3.0)
        return cls._clean(frac / 3.0 * 2.0 - 1.0)                 # 映射到 [-1, 1]

    @classmethod
    def _c_ac2(cls, raw: dict) -> torch.Tensor:
        """二階自相關（lag=2），clip[-1,1]。

        同 _ac1 邏輯但延遲為2：unfold(w+2)，x=wnd[:,:,:-2]，y=wnd[:,:,2:]。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret   = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret   = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), ret], dim=1)
        w     = 20   # 相關窗口長度
        lag   = 2
        pad   = torch.zeros(N, w + lag - 1, device=ret.device, dtype=ret.dtype)
        wnd   = torch.cat([pad, ret], dim=1).unfold(1, w + lag, 1)   # [N, T, w+lag]
        x     = wnd[:, :, :-lag]    # 前 w 個（無 lag 版）
        y     = wnd[:, :,  lag:]    # 後 w 個（lag 版）
        xm    = x.mean(dim=-1, keepdim=True)
        ym    = y.mean(dim=-1, keepdim=True)
        cov   = ((x - xm) * (y - ym)).mean(dim=-1)
        sx    = ((x - xm) ** 2).mean(dim=-1).sqrt()
        sy    = ((y - ym) ** 2).mean(dim=-1).sqrt()
        corr  = cov / (sx * sy + 1e-8)
        return cls._clean(torch.clamp(corr, -1.0, 1.0))

    @classmethod
    def _c_ret_entropy_20(cls, raw: dict) -> torch.Tensor:
        """收益符號的20期滾動香農熵（三分箱：正/負/零），歸一化到[0,1]。

        因果 unfold(20)，每窗口內統計正/負/零收益比例，
        H = -sum(p * log(p+eps))，歸一化 / log(3) ∈ [0, 1]。
        """
        import math
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret   = torch.log(close[:, 1:] / (close[:, :-1] + eps))
        ret   = torch.cat([torch.zeros(N, 1, device=close.device, dtype=close.dtype), ret], dim=1)
        w     = 20
        pad   = torch.zeros(N, w - 1, device=ret.device, dtype=ret.dtype)
        wnd   = torch.cat([pad, ret], dim=1).unfold(1, w, 1)   # [N, T, w]
        # 三分箱比例
        p_pos  = (wnd > 0).float().mean(dim=-1)    # [N, T]
        p_neg  = (wnd < 0).float().mean(dim=-1)
        p_zero = (wnd == 0).float().mean(dim=-1)
        # 香農熵
        def _h(p: torch.Tensor) -> torch.Tensor:
            return -p * torch.log(p + eps)
        H = _h(p_pos) + _h(p_neg) + _h(p_zero)    # ≥0
        return cls._clean(torch.clamp(H / math.log(3), 0.0, 1.0))

    # ── task 5.8 跨截面相對強弱類特徵（補充）compute ──────────────────────

    @classmethod
    def _c_cs_rank_ret5(cls, raw: dict) -> torch.Tensor:
        """5期收益的截面百分位排名（每時間步對 N 品種排名）∈[0,1]。

        N=1 → 0.5；使用 argsort 因果截面排名（不跨時間）。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret5_raw = torch.log(close[:, 5:] / (close[:, :-5] + eps))
        ret5 = torch.cat([torch.zeros(N, 5, device=close.device, dtype=close.dtype), ret5_raw], dim=1)  # [N, T]
        if N == 1:
            return cls._clean(torch.full_like(ret5, 0.5))
        # 截面排名（每時間步沿 N 維）
        order = ret5.argsort(dim=0)                 # [N, T] — argsort 沿品種維
        ranks = torch.zeros_like(ret5)
        ranks.scatter_(0, order, torch.arange(N, dtype=ret5.dtype, device=ret5.device).unsqueeze(1).expand(N, T))
        cs_rank = ranks / (N - 1)                  # ∈ [0, 1]
        return cls._clean(torch.clamp(cs_rank, 0.0, 1.0))

    @classmethod
    def _c_cs_zscore_ret20(cls, raw: dict) -> torch.Tensor:
        """20期收益的截面 z-score（每時間步跨品種去均值/除標準差）。

        N=1 → 0；robust_norm 兜底防極端值。
        """
        eps   = cls._EPS
        close = raw["close"].float()
        N, T  = close.shape
        ret20_raw = torch.log(close[:, 20:] / (close[:, :-20] + eps))
        ret20 = torch.cat([torch.zeros(N, 20, device=close.device, dtype=close.dtype), ret20_raw], dim=1)  # [N, T]
        if N == 1:
            return cls._clean(torch.zeros_like(ret20))
        cs_mean = ret20.mean(dim=0, keepdim=True)        # [1, T]
        cs_std  = ret20.std(dim=0, keepdim=True) + eps   # [1, T]
        zscore  = (ret20 - cs_mean) / cs_std             # [N, T]
        return cls._norm(cls._clean(zscore))

    # ── main ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_features(raw_dict: dict) -> torch.Tensor:
        """按 FEATURE_REGISTRY 註冊順序計算全部特徵，堆疊為 [N, F, T]。

        逐特徵 compute 的數值與順序與重構前逐元素一致；出口統一 nan_to_num→0。
        """
        feats = [spec.compute(raw_dict) for spec in FEATURE_REGISTRY.feature_specs]
        features = torch.stack(feats, dim=1)
        return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ── FEATURE_REGISTRY：現有 30 特徵的聲明式註冊（task 5.1）────────────────
# 順序嚴格對應重構前 compute_features 的 stack 順序（0..29），不可變更。
# category 用於報告分組與類別覆蓋校驗（依 vocab.py 注釋歸類）。

FEATURE_REGISTRY = Registry()

_fe = MT5FeatureEngineer

# (name, category, compute) —— 順序即 token/特徵維順序
_FEATURE_DEFS = [
    # 趨勢類 trend (0-4)
    ("RET",         "trend",           _fe._c_ret),
    ("RET5",        "trend",           _fe._c_ret5),
    ("RET20",       "trend",           _fe._c_ret20),
    ("MA_DIFF",     "trend",           _fe._c_ma_diff),
    ("SLOPE20",     "trend",           _fe._c_slope20),
    # 波動類 volatility (5-8)
    ("ATR",         "volatility",      _fe._c_atr),
    ("RVOL",        "volatility",      _fe._c_rvol),
    ("HL_RANGE",    "volatility",      _fe._c_hl_range),
    ("VOL_REGIME",  "volatility",      _fe._c_vol_regime),
    # 反轉類 reversal (9-13)
    ("DEV",         "reversal",        _fe._c_dev),
    ("DEV60",       "reversal",        _fe._c_dev60),
    ("RSI14",       "reversal",        _fe._c_rsi14),
    ("PRESSURE",    "reversal",        _fe._c_pressure),
    ("AC1",         "reversal",        _fe._c_ac1),
    # 成交量類 volume (14-16)
    ("VOL_RATIO",   "volume",          _fe._c_vol_ratio),
    ("VOL_Z",       "volume",          _fe._c_vol_z),
    ("PV_CORR",     "volume",          _fe._c_pv_corr),
    # 跨截面相對強弱 cross_sectional (17-19)
    ("REL_RET5",    "cross_sectional", _fe._c_rel_ret5),
    ("REL_RET20",   "cross_sectional", _fe._c_rel_ret20),
    ("REL_VOL",     "cross_sectional", _fe._c_rel_vol),
    # v3.0 新增特徵 (20-25)
    ("VWAP_DEV",    "volume",          _fe._c_vwap_dev),
    ("BOLL_POS",    "channel",         _fe._c_boll_pos),
    ("BOLL_WIDTH",  "volatility",      _fe._c_boll_width),
    ("MACD_HIST",   "momentum",        _fe._c_macd_hist),
    ("OBV_SLOPE",   "volume",          _fe._c_obv_slope),
    ("MFI14",       "volume",          _fe._c_mfi14),
    # v3.0 Alpha 101 + 互補特徵 (26-29)
    ("WILLR_14",    "reversal",        _fe._c_willr14),
    ("CCI_14",      "reversal",        _fe._c_cci14),
    ("ROC_12",      "momentum",        _fe._c_roc12),
    ("TYPICAL_DEV", "reversal",        _fe._c_typical_dev),
    # task 5.2 趨勢類 trend (30-32)
    ("EMA_RATIO_12_26",  "trend",      _fe._c_ema_ratio_12_26),
    ("TREND_STRENGTH_50","trend",      _fe._c_trend_strength_50),
    ("PRICE_POS_50",     "trend",      _fe._c_price_pos_50),
    # task 5.2 動量類 momentum (33-36)
    ("TRIX_15",     "momentum",        _fe._c_trix_15),
    ("PPO",         "momentum",        _fe._c_ppo),
    ("ULT_OSC",     "momentum",        _fe._c_ult_osc),
    ("RET_ACCEL",   "momentum",        _fe._c_ret_accel),
    # task 5.3 波動類（含 OHLC 估計量）volatility (37-40)
    ("GK_VOL",          "volatility",  _fe._c_gk_vol),
    ("PARKINSON_VOL",   "volatility",  _fe._c_parkinson_vol),
    ("YANG_ZHANG_VOL",  "volatility",  _fe._c_yang_zhang_vol),
    ("RS_VOL",          "volatility",  _fe._c_rs_vol),
    # task 5.4 量能/流動性類 volume (41-44)
    ("AMIHUD_ILLIQ",    "volume",      _fe._c_amihud_illiq),
    ("KYLE_LAMBDA",     "volume",      _fe._c_kyle_lambda),
    ("CMF_20",          "volume",      _fe._c_cmf_20),
    ("AD_LINE_SLOPE",   "volume",      _fe._c_ad_line_slope),
    # task 5.5 反轉/振盪類 reversal/trend/momentum (45-50)
    ("STOCH_K_14",      "reversal",    _fe._c_stoch_k_14),
    ("STOCH_D_3",       "reversal",    _fe._c_stoch_d_3),
    ("AROON_OSC_25",    "reversal",    _fe._c_aroon_osc_25),
    ("DMI_ADX_14",      "trend",       _fe._c_dmi_adx_14),
    ("DMI_DIFF_14",     "trend",       _fe._c_dmi_diff_14),
    ("TRIX_SIGNAL",     "momentum",    _fe._c_trix_signal),
    # task 5.6 通道/突破類 channel (51-56)
    ("DONCHIAN_POS_20",     "channel", _fe._c_donchian_pos_20),
    ("KELTNER_POS_20",      "channel", _fe._c_keltner_pos_20),
    ("ICHIMOKU_KIJUN_DEV",  "channel", _fe._c_ichimoku_kijun_dev),
    ("ICHIMOKU_TENKAN_DEV", "channel", _fe._c_ichimoku_tenkan_dev),
    ("SUPERTREND_DIR",      "channel", _fe._c_supertrend_dir),
    ("SAR_DIST",            "channel", _fe._c_sar_dist),
    # task 5.7 統計類 statistical (57-62)
    ("ROLL_SKEW_20",    "statistical", _fe._c_roll_skew_20),
    ("ROLL_KURT_20",    "statistical", _fe._c_roll_kurt_20),
    ("HURST_50",        "statistical", _fe._c_hurst_50),
    ("FRACTAL_DIM_30",  "statistical", _fe._c_fractal_dim_30),
    ("AC2",             "statistical", _fe._c_ac2),
    ("RET_ENTROPY_20",  "statistical", _fe._c_ret_entropy_20),
    # task 5.8 跨截面相對強弱補充 cross_sectional (63-64)
    ("CS_RANK_RET5",    "cross_sectional", _fe._c_cs_rank_ret5),
    ("CS_ZSCORE_RET20", "cross_sectional", _fe._c_cs_zscore_ret20),
]

# ── 啟用特徵白名單（特徵剪枝機制，2026-07-04）──────────────────────────
# 若項目根目錄存在 active_features.json（由 prune_features.py 生成），
# 則只註冊白名單內的特徵，縮小 vocab 與搜索空間；否則註冊全部 65 個特徵。
# 白名單按 _FEATURE_DEFS 原始順序過濾，保持特徵維順序穩定。
def _load_active_feature_allowlist() -> set[str] | None:
    import json as _json
    import pathlib as _pathlib
    _path = _pathlib.Path(__file__).resolve().parent.parent / "active_features.json"
    if not _path.exists():
        return None
    try:
        data = _json.loads(_path.read_text(encoding="utf-8"))
        names = data.get("active_features") if isinstance(data, dict) else data
        allow = {str(n) for n in names}
        return allow or None
    except Exception:
        return None


_ACTIVE_FEATURES = _load_active_feature_allowlist()

for _name, _category, _compute in _FEATURE_DEFS:
    if _ACTIVE_FEATURES is not None and _name not in _ACTIVE_FEATURES:
        continue
    FEATURE_REGISTRY.register_feature(
        FeatureSpec(name=_name, category=_category, compute=_compute)
    )

# 由註冊表導出有序特徵名視圖（保持 import 相容；vocab.py 側整合見後續任務）
FEATURE_NAMES = FEATURE_REGISTRY.feature_names

# 計數一致性（R1.13）：INPUT_DIM == len(FEATURE_NAMES) == F
MT5FeatureEngineer.INPUT_DIM = len(FEATURE_REGISTRY.feature_names)
