"""
model_core/config.py — 模型層配置

僅保留模型訓練所需的參數。
品種、數據、風控等全局配置統一由根目錄 config.py 的 Config 類管理。
"""
import math

import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    # ── 訓練設備 ─────────────────────────────────────────────────────────
    # 注意：本任務 CPU 訓練速度反而比 GPU 快（實測約 2.3 倍），故強制用 CPU。
    # 原因：
    #   1. 張量太小——forex 組僅 (2 品種 × 3508 × 20 特徵)，單個運算元的
    #      計算量小於 CUDA kernel 啟動開銷（數十微秒），GPU 算得快但啟動慢。
    #   2. 訓練循環是 Python 串列調度：每 step 逐條跑 128 條公式 × 8 個
    #      VM 步 × 4 個 walk-forward 折，GPU 被切成上萬個碎片段，吃不滿。
    #   3. host↔device 拷貝 + kernel 啟動延遲主導總耗時，而非張量計算本身。
    #   4. 實測 GPU 利用率 ~51%，正是 GPU 一半時間在乾等 Python 餵下一個
    #      kernel 的證據（不是“還能壓榨”，而是“調度瓶頸”）。
    # 基準測試（forex 組, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若後續改為批次並行公式評估（一次餵大批張量進 GPU），再切回 cuda。
    DEVICE = torch.device("cpu")

    # ── 訓練參數（大搜索空間適配版，2026-07-04 重構）─────────────────────
    # 背景：特徵庫擴展到 65、運算元庫擴展到 66（vocab=131），8-token 搜索空間
    #   從舊版 ~7億 暴增到 ~8.67×10^16（1.2 億倍）。舊的採樣預算（128×3000）
    #   覆蓋率趨近於零，導致熵坍塌 Early Stop、公式退化。
    # 對策（訓練時間不敏感場景）：
    #   1. 特徵剪枝（active_features.json）把 vocab 降到 ~90，空間縮小約 20 倍
    #   2. 放大採樣預算：BATCH_SIZE 128→256，TRAIN_STEPS 3000→8000
    #   3. 更大精英池（60）保留更多歷史最優
    BATCH_SIZE      = 192   # 每步採樣公式數（原 128，1.5x 提升覆蓋率）
    TRAIN_STEPS     = 9000  # 每組訓練步數（55次重啟需要更多步數）
    MAX_FORMULA_LEN = 8     # 公式長度上限：保持 8（10 會導致 CPU 訓練慢 3 倍）

    # ── 特徵維度（由 vocab.py 自動派生，無需手動修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 為主，IC 做門控 ──────────────────────────────────
    # IC_NEG_MULT 0.30→0.50：0.30 對反向因子懲罰過重，可能誤殺非線性高收益因子。
    # 收益優先模式下，只要年化收益是正的，適當負 IC 可以接受。
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.01
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.75   # 收益優先：不過度誤殺反向/非線性高收益因子

    # ── FTMO 專屬獎勵模式 ─────────────────────────────────────────────
    # "standard": 收益+風險平衡（默認，原權重）
    # "ftmo":     FTMO 考試盤專屬——年化收益權重 0.60→0.75，Calmar 0.05→0.10
    #             （控制 MDD 貼近 10% Max Loss 上限），其餘指標權重下調。
    #             目標：在 10% Max Loss 約束下最大化年化收益，快速達標。
    # "forex":    外匯均值回歸專屬（2026-07-08）——
    #             降年化收益權重(0.80→0.25)、提IC權重(0.03→0.25)、
    #             新增反轉獎勵(0.20，獎勵低/負因子自相關)和多空對稱檢查(0.15)。
    #             原因：外匯H1以震盪為主，趨勢運算元效果差，需引導模型偏好
    #             均值回歸信號而非追漲殺跌。
    REWARD_MODE:       str = "ftmo"

    # ── 熵保護（大空間加強版）──────────────────────────────────────────
    # ENTROPY_COEFF_MAX 0.5→1.0：加倍探索壓力，對抗大 vocab 的過早收斂。
    # ENTROPY_COLLAPSE_THRESH 改為相對閾值 0.15×ln(vocab)：大 vocab 最大熵更高
    #   （ln(131)≈4.87 vs ln(54)≈3.99），絕對閾值 0.5 不再合理。
    # ENTROPY_COLLAPSE_STEPS 15→40：給模型更長的自我恢復窗口，不急於重啟。
    ENTROPY_COEFF_MAX:   float = 1.0
    ENTROPY_COEFF_POWER: float = 1.0  # 降低冪次，讓低熵時係數更激進（原1.3）
    ENTROPY_COLLAPSE_THRESH: float = 0.15 * math.log(FORMULA_VOCAB.size)
    ENTROPY_COLLAPSE_STEPS:  int   = 20  # 更快檢測坍塌並重啟

    # ── 熵下限懲罰（Fix 1: H→0 時熵項歸零問題）──────────────────────────
    # 當 H < ENTROPY_FLOOR_THRESH 時，加入固定懲罰 λ×(thresh-H)。
    # 這確保即使 mean_ent→0，loss 中仍有非零探索壓力。
    ENTROPY_FLOOR:        bool  = True
    ENTROPY_FLOOR_THRESH: float = 1.0   # 熵低於此值時觸發固定懲罰（提高介入時機）
    ENTROPY_FLOOR_LAMBDA: float = 5.0   # 懲罰強度係數（加大力度對抗坍塌）

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25
    ELITE_POOL_SIZE:    int   = 60    # 30→60：大空間需要更大的精英記憶
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重啟（大空間加強版）─────────────────────────────────────────
    # MAX_RESTARTS 8→25→55、RESTART_NOISE 0.05→0.1→0.25：時間不敏感，多給機會+更強擾動。
    # 配合 engine.py：超過 MAX_RESTARTS 後不再 Early Stop，改為強擾動繼續訓練。
    # 2026-07-09: US100 訓練 24/25 重啟仍有突破，擴到 55 次。
    MAX_RESTARTS:   int   = 55
    RESTART_NOISE:  float = 0.25

    # ── 自適應噪聲：Best 停滯時自動增大擾動 ─────────────────────────────
    # stagnation_window: 判斷停滯的步數窗口
    # noise_min / noise_max: 噪聲下界和上界
    # noise_boost: 停滯時噪聲提升倍率
    ADAPTIVE_NOISE:      bool  = True
    STAGNATION_WINDOW:   int   = 500
    NOISE_MIN:           float = 0.15
    NOISE_MAX:           float = 0.60
    NOISE_BOOST_FACTOR:  float = 2.0   # noise += 0.2 * (stagnation / window)

    # ── 重啟多樣性（Fix 2: best_snapshot 吸引子效應）─────────────────────
    # 每 FULL_RESET_EVERY 次重啟中，做 1 次完全隨機初始化而非從 best_snapshot 恢復。
    FULL_RESET_EVERY:    int   = 3     # 每 3 次重啟中第 3 次做 full reset

    # ── Reward baseline（Fix 3: 全負 batch 相對優選問題）──────────────────
    # 用 EMA baseline 替代 batch mean 計算 advantage，避免全負 batch 的問題。
    REWARD_EMA_BASELINE:     bool  = True
    REWARD_EMA_DECAY:        float = 0.95   # EMA 衰減係數
    REWARD_EMA_WARMUP:       int   = 10     # 前 N 步用 batch mean（EMA 未穩定）

    # ── 重啟時部分重設參數：保留底層，擾動頂層 ───────────────────────────
    PARTIAL_RESET:       bool  = True
    PARTIAL_RESET_LAYERS: tuple = ("ln_f", "mtp_head", "head_critic", "blocks", "token_emb")

    # ── Elite Replay 衰減：舊 elite 採樣權重隨時間衰減 ──────────────────
    ELITE_DECAY:         bool  = True
    ELITE_DECAY_HALF_LIFE: int = 300   # 每 300 步舊 elite 權重減半

    # ── 多起點並行（Island）──────────────────────────────────────────────
    # 注意：Island 模式在 CPU 訓練下會讓總時間變成 N 倍（islands 串列），
    # 對於 index 這類大數組（T=32076）會變得極慢。當前默認關閉，保留配置開關。
    N_ISLANDS:              int   = 1
    MIGRATION_INTERVAL:     int   = 500
    MIGRATION_TOP_K:        int   = 5
    # island 默認關閉，避免用戶誤開導致速度爆炸

    # ── 因子去相關參數 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 25
    CORR_THRESHOLD:   float = 0.85
    CORR_PENALTY:     float = 0.8

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    WF_GAP: int = 20

    # ── 公式結構約束（2026-07-05 新增）──────────────────────────────────
    # 背景：index 組因子因 TS_RANK 連續使用退化為 beta 因子（91.8% 做多），
    # 前半段市場跌虧錢、後半段市場漲賺錢，不是 alpha 而是 beta。
    # 對策：在採樣階段禁止恆正運算元鏈，在評分階段添加 beta 中性 + 前後一致性獎懲。
    ENABLE_FORMULA_STRUCTURE_CONSTRAINT: bool = True   # 總開關
    BETA_NEUTRAL_PENALTY:     bool  = True             # 多空比例失衡懲罰
    HALF_CONSISTENCY_BONUS:   bool  = True             # 前後一致性獎懲
    BETA_NEUTRAL_THRESH:      float = 0.85             # 超過此比例同方向觸發重罰
    BETA_NEUTRAL_LIGHT_THRESH: float = 0.70            # 輕度失衡閾值
