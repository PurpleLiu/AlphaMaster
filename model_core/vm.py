import torch
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB

# ── 恆正運算元集 ────────────────────────────────────────────────────────────
# 這些運算元輸出值域非負（或幾乎恆正），連續使用會遺失符號資訊，
# 導致因子退化成「永遠做多」的 beta 因子。
# TS_RANK_*: 輸出 [0, 1)，永遠非負
# ABS: 取絕對值，遺失符號
# TS_SUM_*: 本身不恆正，但如果輸入已經非負則輸出更正
# TS_MAX_*: 取最大值，如果輸入含正數則偏向正
# 我們用「感染性運算元」概念：一旦前面出現了恆正運算元，後續的 TS_SUM/TS_MEAN/TS_MAX
# 都不會讓值域變回有正有負，反而會強化正值。只有 SUB/NEG/DIV/TS_ZSCORE/CS_NEUTRALIZE
# 等運算元才能「恢復」符號資訊。
POSITIVE_ONLY_OPS = {"TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS"}
# 感染傳播運算元：在恆正值域上使用時，輸出仍為恆正
# TS_SUM_*: 非負值的和仍非負
# TS_MEAN_*: 非負值的均值仍非負
# TS_MAX_*: 非負值的最大值仍非負
# CLIP: clamp(-3,3) 不改變符號，但如果輸入全非負則輸出也全非負
# SQRT: sign(x)*sqrt(|x|)，如果輸入全非負則輸出全非負
# POWER: sign(x)*|x|^2，如果輸入全非負則輸出全非負
# SIGNED_LOG: sign(x)*log1p(|x|)，如果輸入全非負則輸出全非負
# SIGMOID: 2*sigmoid-1，對非負輸入輸出正值
# TANH_SQUASH: tanh，對非負輸入輸出正值
INFECTED_PROPAGATING_OPS = {
    "TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS",
    "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
    "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
    "TS_MAX_10", "TS_MAX_20",
    "CLIP", "SQRT", "POWER", "SIGNED_LOG",
    "SIGMOID", "TANH_SQUASH", "WINSORIZE",
    "WMA", "EMA_5", "EMA_20", "DECAY", "DECAY_LINEAR_5",
    "TS_DECAY_EXP_5",
}
# 恢復運算元：能夠把恆正值域重新變成有正有負
SIGN_RESTORE_OPS = {
    "SUB", "DIV", "NEG", "GATE", "IF_GT",
    "TS_ZSCORE_10", "TS_ZSCORE_20",
    "CS_NEUTRALIZE", "CS_RANK", "CS_SCALE",
    "TS_STD_5", "TS_STD_10", "TS_STD_20",
    "TS_CORR_10", "TS_SKEW_10", "TS_QUANTILE_10",
    "DELTA", "DELTA_5", "MOMENTUM_5", "MOMENTUM_10",
    "PPO",  # 但 PPO 是特徵不是運算元
}


def is_positive_only_op(token_name: str) -> bool:
    """判斷運算元是否輸出恆正值（可能遺失符號資訊）。"""
    return token_name in POSITIVE_ONLY_OPS


def is_infected_propagating(token_name: str) -> bool:
    """判斷運算元是否會傳播恆正感染（在恆正輸入上輸出仍恆正）。"""
    return token_name in INFECTED_PROPAGATING_OPS


def is_sign_restoring(token_name: str) -> bool:
    """判斷運算元是否能恢復符號資訊（把恆正值域變回有正有負）。"""
    return token_name in SIGN_RESTORE_OPS


def validate_formula_structure(formula_tokens: list[int], vocab_names: tuple[str, ...]) -> list[str]:
    """校驗公式結構，返回違規原因列表（空列表 = 合法）。
    
    使用「感染模型」：一旦公式中出現恆正運算元（如 TS_RANK），
    後續如果連續使用傳播運算元（如 TS_SUM/TS_MEAN/CLIP/SQRT），
    值域會一直保持非負，導致因子退化成 beta。
    只有恢復運算元（如 SUB/TS_ZSCORE/CS_NEUTRALIZE）才能打破感染。
    
    規則：
    1. 禁止恆正運算元後連續 2 個以上傳播運算元（感染鏈太長）
    2. 公式末尾如果是感染狀態（恆正且未恢復），標記違規
    """
    violations = []
    feat_offset = FORMULA_VOCAB.operator_offset
    
    infected = False  # 當前值域是否已被感染（恆正）
    infected_chain_len = 0  # 感染鏈長度
    last_positive_op = None  # 最後一個引發感染的運算元名
    
    for i, token in enumerate(formula_tokens):
        token = int(token)
        if token < feat_offset:
            # 特徵 token：不改變感染狀態
            continue
        name = vocab_names[token] if token < len(vocab_names) else f"op_{token}"
        
        if is_positive_only_op(name):
            # 恆正運算元：開始/延續感染
            if not infected:
                infected = True
                last_positive_op = name
            infected_chain_len += 1
            
        elif infected and is_sign_restoring(name):
            # 恢復運算元：打破感染
            infected = False
            infected_chain_len = 0
            last_positive_op = None
            
        elif infected and is_infected_propagating(name):
            # 傳播運算元：感染繼續
            infected_chain_len += 1
            # 規則1：感染鏈超過 3 個運算元時報警
            if infected_chain_len >= 3:
                violations.append(
                    f"步驟{i}: 恆正感染鏈過長（從 {last_positive_op} 起 {infected_chain_len} 個傳播運算元），"
                    f"因子將退化為 beta"
                )
        # else: 非感染相關運算元（如 ADD/MUL），不改變感染狀態
    
    # 規則2：公式末尾處於感染狀態
    if infected and infected_chain_len >= 2:
        violations.append(
            f"公式末尾處於恆正感染狀態（鏈長 {infected_chain_len}），"
            f"因子輸出將偏向單方向"
        )
    
    return violations

# ── 擴展後詞表規模說明（task 12.1）──────────────────────────────────────────
#
# 本次擴展（factor-operator-library-expansion）後：
#   - 特徵數 F  = len(FORMULA_VOCAB.feature_names)  （當前 65，覆蓋 8 大類）
#   - 運算元數 O  = len(OPS_CONFIG)                    （當前 66）
#   - 詞表總 size = F + O = 131
#   - feat_offset = F = 65（feature token id ∈ [0, 64]）
#   - operator token id ∈ [F, F+O-1] = [65, 130]
#
# StackVM 的 op_map / arity_map **完全動態**從 FORMULA_VOCAB 與 OPS_CONFIG 派生，
# 不寫死任何 token 數或偏移值，因此無需在此處做任何結構變更。後續再次擴展
# 特徵或運算元時只需更新註冊表，VM 自動消費。
#
# Cross-sectional 運算元（CS_RANK / CS_SCALE / CS_NEUTRALIZE）沿 N 維逐時間步
# 操作，輸入/輸出形狀均為 [N, T]，滿足統一的 [N,T]→[N,T] 契約（R2.9）；
# VM 主循環無需對它們做任何特殊處理。


class StackVM:
    def __init__(self):
        # feat_offset 動態從 FORMULA_VOCAB.operator_offset 讀取（= feature_count = F）。
        self.feat_offset = FORMULA_VOCAB.operator_offset
        # op_map / arity_map 動態從 OPS_CONFIG 構建。
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        # 恆正運算元 token id 集合（用於採樣時約束）
        self.positive_only_ids = set()
        for i, cfg in enumerate(OPS_CONFIG):
            if cfg[0] in POSITIVE_ONLY_OPS:
                self.positive_only_ids.add(i + self.feat_offset)

    @staticmethod
    def _normalize_output(x: torch.Tensor) -> torch.Tensor:
        """
        對因子輸出做標準化，確保幅度足夠觸發 neutral band 入場。

        策略（三級降級）：
        1. 截面 zscore（跨品種，每時間步）：適合因子跨品種有分散
        2. 時序 zscore（每品種，全局）：當截面 std 太小時使用
        3. 若兩級都失敗（因子是常數）：返回原值，由 const_cnt 攔截

        Returns:
            [N, T] clip 到 [-3, 3]，若是常數則返回原值（engine 會過濾）
        """
        N, T = x.shape

        # 檢測是否是全局常數（標準化無意義）
        global_std = x.std()
        if global_std < 1e-6:
            return x   # 常數因子，由 engine 的 const_cnt 攔截

        # ── 截面標準化（跨品種，每時間步；N=1 時跳過）──────────────
        if N > 1:
            cs_mean = x.mean(dim=0, keepdim=True)
            cs_std  = x.std(dim=0, keepdim=True).clamp(min=1e-8)
            cs_z    = (x - cs_mean) / cs_std
            if cs_z.std() >= 0.3:
                return torch.clamp(cs_z, -3.0, 3.0)

        # ── 時序標準化（每品種獨立，expanding 無 look-ahead）─────────
        # 每個 t 僅用 x[:, :t+1] 計算 mean/std，避免用 t 之後的未來統計量
        # 歸一化當前值（原 dim=1 全局 mean/std 存在 look-ahead bias）。
        cnt = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T)
        cumsum = x.cumsum(dim=1)
        ts_mean = cumsum / cnt                          # [N,T]，t 位 = x[:,:t+1].mean()
        cumsum_sq = (x * x).cumsum(dim=1)
        ts_var = (cumsum_sq / cnt) - ts_mean * ts_mean  # E[x^2] - E[x]^2
        ts_std = ts_var.clamp(min=1e-8).sqrt()
        ts_z = (x - ts_mean) / ts_std

        if ts_z.std() >= 0.1:
            return torch.clamp(ts_z, -3.0, 3.0)

        # ── 兩級均失敗：因子無區分度，返回原值讓 engine 過濾 ────────
        return x

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                result = stack[0]
                # 最終輸出標準化：保證因子幅度足夠，避免全程空倉
                return self._normalize_output(result)
            else:
                return None
        except Exception:
            return None
