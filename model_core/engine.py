import copy
import heapq
import json
import math
import pathlib
import random
import sys

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MT5Backtest, estimate_periods_per_year
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2

# P3：冠軍在場時間穩健性校驗所需
try:
    from strategy_manager.signal import compute_target_positions_stateless
except ImportError:
    # 相容無 strategy_manager 的測試環境
    def compute_target_positions_stateless(factors):  # type: ignore
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE  = _RootConfig.STRATEGY_FILE
    _CHECKPOINT_DIR = pathlib.Path(getattr(_RootConfig, 'CHECKPOINT_DIR', 'checkpoints'))
except ImportError:
    _STRATEGY_FILE  = "best_mt5_strategy.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _strategy_file_for_symbol(symbol: str | None) -> str:
    """返回該品種對應的策略文件路徑。

    單品種訓練時使用 strategies/best_{symbol}.json，
    多品種/未指定品種時回退到默認路徑。
    """
    if symbol:
        return str(pathlib.Path("strategies") / f"best_{symbol}.json")
    return _STRATEGY_FILE


def _fallback_data_file_for_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Read web_settings.json last_data_file when strategy JSON lacks data_file."""
    settings_path = pathlib.Path("web_settings.json")
    if not settings_path.exists():
        return None, None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        last = str(settings.get("last_data_file") or "").strip()
    except (json.JSONDecodeError, OSError):
        return None, None
    if not last:
        return None, None
    p = pathlib.Path(last)
    if not p.exists():
        return None, None
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(p.resolve()))
    except Exception:
        return str(p.resolve()), None
    if info.get("symbol") != symbol:
        return None, None
    return str(p.resolve()), info.get("timeframe")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 摺疊構建
# ─────────────────────────────────────────────────────────────────────────────

def _build_walk_forward_folds(T: int, n_folds: int = 5, gap: int = 20) -> list[dict]:
    fold_size = T // n_folds
    if fold_size < 2:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    total_required = fold_size * n_folds + gap * (n_folds - 1)
    if total_required > T:
        gap = max(0, (T - fold_size * n_folds) // n_folds)
    folds = []
    for k in range(1, n_folds):
        train_end = fold_size * k
        val_start = train_end + gap
        val_end   = val_start + fold_size if k < n_folds - 1 else T
        if val_start >= T or val_end > T:
            break
        folds.append({"train_start": 0, "train_end": train_end,
                      "val_start": val_start, "val_end": val_end, "gap": gap})
    if not folds:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    return folds


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保證 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedSampler:
    def __init__(self, vocab_size: int, feat_offset: int, arity_map: dict[int, int],
                 positive_only_ids: set[int] | None = None):
        self.vocab_size  = vocab_size
        self.feat_offset = feat_offset
        self.arity_map   = arity_map
        self.delta: dict[int, int] = {}
        for tid in range(vocab_size):
            if tid < feat_offset:
                self.delta[tid] = 1
            else:
                a = arity_map.get(tid, 1)
                self.delta[tid] = 1 - a
        # 恆正運算元 token id 集合（用於運算元鏈約束）
        self.positive_only_ids = positive_only_ids or set()
        # 構建感染傳播/恢復運算元 id 集合
        from .vm import INFECTED_PROPAGATING_OPS, SIGN_RESTORE_OPS
        from .ops import OPS_CONFIG as _ops
        self.infected_propagating_ids = set()
        self.sign_restore_ids = set()
        for i, cfg in enumerate(_ops):
            tid = i + feat_offset
            if cfg[0] in INFECTED_PROPAGATING_OPS:
                self.infected_propagating_ids.add(tid)
            if cfg[0] in SIGN_RESTORE_OPS:
                self.sign_restore_ids.add(tid)

    def valid_mask(self, stack_depth: int, step_idx: int,
                   total_steps: int, device: torch.device,
                   prev_token: int | None = None,
                   infected_chain_len: int = 0) -> torch.Tensor:
        remaining = total_steps - step_idx
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        for tid in range(self.vocab_size):
            d         = self.delta[tid]
            new_depth = stack_depth + d
            if new_depth < 1:
                mask[tid] = False;  continue
            min_future = new_depth + (remaining - 1) * (-2)
            max_future = new_depth + (remaining - 1) * 1
            if 1 < min_future or 1 > max_future:
                mask[tid] = False
            # ── 運算元鏈約束（感染模型）──────────────────────────────
            # 如果已感染且感染鏈 >= 2，禁止再使用傳播運算元
            # （允許恢復運算元和非傳播運算元如 ADD/SUB/MUL）
            if infected_chain_len >= 2 and tid in self.infected_propagating_ids:
                mask[tid] = False
            # 如果已感染且感染鏈 >= 3，禁止所有運算元（強制恢復或結束）
            # 實際上不禁止恢復運算元，只禁止傳播和恆正運算元
            if infected_chain_len >= 3:
                if tid in self.infected_propagating_ids or tid in self.positive_only_ids:
                    mask[tid] = False
        if not mask.any():
            for tid in range(self.vocab_size):
                if stack_depth + self.delta[tid] >= 1:
                    mask[tid] = True
        return mask

    def apply_mask_to_logits(self, logits: torch.Tensor, stack_depths: list[int],
                              step_idx: int, total_steps: int,
                              prev_tokens: list[int | None] | None = None,
                              infected_chain_lens: list[int] | None = None) -> torch.Tensor:
        masked = logits.clone()
        device = logits.device
        for b, depth in enumerate(stack_depths):
            prev_t = prev_tokens[b] if prev_tokens else None
            icl = infected_chain_lens[b] if infected_chain_lens else 0
            vmask = self.valid_mask(depth, step_idx, total_steps, device,
                                    prev_token=prev_t, infected_chain_len=icl)
            masked[b][~vmask] = -1e9
        return masked

    def update_infection(self, token: int, infected_chain_len: int) -> int:
        """更新感染鏈長度，返回新的感染鏈長度。"""
        if token in self.positive_only_ids:
            return infected_chain_len + 1
        elif token in self.sign_restore_ids:
            return 0
        elif token in self.infected_propagating_ids:
            if infected_chain_len > 0:
                return infected_chain_len + 1
            return 0
        return infected_chain_len  # 非傳播/非恢復運算元，不改變狀態


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEngine — __init__ 與靜態輔助方法
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5, n_folds: int = 5,
                 target_symbol: str | None = None):
        self.data_manager  = data_manager
        self.n_folds       = n_folds
        self.target_symbol = target_symbol   # None = 多品種模式，str = 單品種模式
        self.model   = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["in_proj", "out_proj", "qk_norm"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MT5Backtest()

        from .vocab import FORMULA_VOCAB as _v
        self.sampler = ConstrainedSampler(
            vocab_size=_v.size, feat_offset=_v.operator_offset,
            arity_map=self.vm.arity_map,
            positive_only_ids=self.vm.positive_only_ids
        )

        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [], 'stable_rank': []
        }
        self._restart_count      = 0
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0

        # 自適應噪聲：記錄 best 刷新步數
        self._best_update_step = 0
        self._stagnation_steps = 0

        # Fix 3: EMA reward baseline
        self._reward_ema: float | None = None
        self._reward_ema_step: int = 0

    # ── IC computation ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ic(factor: torch.Tensor, target_ret: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """時序 IC（每品種內部 factor[t] vs ret[t+1]）的均值與穩定性。

        對 5 品種宇宙，時序 IC 比橫截面 IC 統計意義更強。
        """
        N, T = factor.shape
        if T < 2:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_list = []
        for n in range(N):
            x  = factor[n, :-1]
            y  = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_tensor = torch.stack(ic_list)
        ic_mean   = ic_tensor.mean()
        ic_stab   = (ic_mean / (ic_tensor.std(unbiased=False) + 1e-6)
                     if ic_tensor.numel() >= 2
                     else torch.zeros(1, device=factor.device))
        return ic_mean, ic_stab

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """IC 門控：用 IC 符號而非量值調整 reward，完全規避量綱問題。
        IC > thresh  → reward × IC_GATE_MULT  (正向預測，獎勵)
        IC < -thresh → reward × IC_NEG_MULT   (反向預測，懲罰)
        |IC| ≤ thresh→ 不修改                  (噪聲區)
        """
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return reward * ModelConfig.IC_GATE_MULT
        elif ic_val < -t:
            return reward * ModelConfig.IC_NEG_MULT
        return reward


    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        """對精英池去重：相同 tokens 只保留得分最高的一條，重建最小堆。"""
        best: dict[str, tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = str(toks)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, cnt, toks, birth)
        deduped = list(best.values())
        heapq.heapify(deduped)
        return deduped

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        """維護精英公式池（最小堆，Top-ELITE_POOL_SIZE 個歷史最優公式，自動去重）。

        去重邏輯：若 formula 已在池中，只在新得分更高時原地更新，不插入重複副本。
        這防止了單一公式壟斷 elite pool，保持多樣性。
        新增：記錄 birth_step 用於 elite decay。
        """
        k = ModelConfig.ELITE_POOL_SIZE

        # 檢查是否已有相同公式
        for idx, (sc, cnt, toks, birth) in enumerate(self._elite_pool):
            if toks == formula:
                if val_score <= sc:
                    return  # 已有更高分的相同公式，不更新
                # 分數更高：從堆中移除舊條目，插入新條目
                self._elite_pool[idx] = self._elite_pool[-1]
                self._elite_pool.pop()
                heapq.heapify(self._elite_pool)  # O(k)，k≤20，可接受
                break

        entry = (val_score, self._elite_counter, list(formula), step)
        self._elite_counter += 1
        if len(self._elite_pool) < k:
            heapq.heappush(self._elite_pool, entry)
        elif val_score > self._elite_pool[0][0]:
            heapq.heapreplace(self._elite_pool, entry)

    # ── Factor pool ───────────────────────────────────────────────────────────

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, self._factor_pool_counter, f_gpu)
        self._factor_pool_counter += 1
        if len(self.factor_pool) < k:
            heapq.heappush(self.factor_pool, entry)
        elif val_score > self.factor_pool[0][0]:
            heapq.heapreplace(self.factor_pool, entry)

    def _apply_corr_penalty(self, reward: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        if not self.factor_pool:
            return reward
        f_flat = factor.detach().reshape(-1).float()
        if f_flat.std() < 1e-4:
            return reward
        pool_vecs = torch.stack(
            [f.reshape(-1).float() for _, _cnt, f in self.factor_pool], dim=0
        )
        f_c  = f_flat - f_flat.mean()
        p_c  = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov  = (p_c * f_c).sum(dim=1)
        sx   = f_c.norm() + 1e-8
        sy   = p_c.norm(dim=1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward * ModelConfig.CORR_PENALTY
        return reward

    def _distribution_stats(self, prev_dist=None):
        """計算模型初始位置（zero prefix）token 分布的細化指標，用於判斷 H 不變時
        分布是否真的在變化。
        """
        vocab_size = FORMULA_VOCAB.size
        with torch.no_grad():
            inp = torch.zeros((1, 1), dtype=torch.long,
                              device=ModelConfig.DEVICE)
            logits, _, _ = self.model(inp)
            logits = self.sampler.apply_mask_to_logits(
                logits, [0], 0, ModelConfig.MAX_FORMULA_LEN
            )
            dist = F.softmax(logits, dim=-1).squeeze(0)
            ent = -(dist * torch.log(dist + 1e-12)).sum().item()
            log_v = math.log(vocab_size)
            kl_uniform = log_v - ent
            top1 = dist.max().item()
            top5 = dist.topk(5, dim=-1).values.sum().item()
            eff_vocab = math.exp(ent)
            prob_std = dist.std(unbiased=False).item()
            kl_prev = 0.0
            if prev_dist is not None:
                kl_prev = (
                    dist * (torch.log(dist + 1e-12) -
                            torch.log(prev_dist.to(dist.device) + 1e-12))
                ).sum().item()
        return {
            'dist': dist.cpu(),
            'entropy': ent,
            'kl_uniform': kl_uniform,
            'top1_prob': top1,
            'top5_prob': top5,
            'eff_vocab': eff_vocab,
            'prob_std': prob_std,
            'kl_prev': kl_prev,
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        if self.data_manager is None:
            raise RuntimeError("AlphaEngine requires a data_manager.")

        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS

        if verbose_header:
            print("開始 Alpha 因子挖掘訓練" +
                  ("（含 LoRD 正則化）..." if self.use_lord else "..."))
            print(f"   策略熵: 坍塌閾值={ModelConfig.ENTROPY_COLLAPSE_THRESH}  "
                  f"係數上限={ModelConfig.ENTROPY_COEFF_MAX}  "
                  f"連續坍塌步數={ModelConfig.ENTROPY_COLLAPSE_STEPS}")
            print(f"   精英重播: 比例={ModelConfig.ELITE_REPLAY_FRAC}  "
                  f"池大小={ModelConfig.ELITE_POOL_SIZE}")
            print(f"   IC門控: 閾值±{ModelConfig.IC_GATE_THRESH}  "
                  f"正向×{ModelConfig.IC_GATE_MULT}  負向×{ModelConfig.IC_NEG_MULT}")
            print(f"   最大重啟: {ModelConfig.MAX_RESTARTS}  "
                  f"噪聲={ModelConfig.RESTART_NOISE}")

        T     = self.data_manager.target_ret.shape[1]
        folds = _build_walk_forward_folds(T, self.n_folds,
                                          gap=getattr(ModelConfig, 'WF_GAP', 20))
        use_wf = len(folds) > 1 and not (
            folds[0]["train_start"] == 0 and folds[0]["train_end"] == T
        )
        if verbose_header:
            if use_wf:
                print(f"   滾動驗證: {len(folds)} 折  共 {T} 根K線")
                for k, f in enumerate(folds):
                    print(f"  第{k+1}折: 訓練[0,{f['train_end']}) "
                          f"間隔={f['gap']} 驗證[{f['val_start']},{f['val_end']})")
            else:
                print(f"   退化為全量評估（共 {T} 根K線）")

        # 因果安全：features.py 的 _robust_norm 已改為滾動因果實現
        # 每個 t 的歸一化參數只用 [t-w+1..t]，walk-forward 摺疊切片無洩露
        feat  = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)

        # 數據驅動年化因子：按訓練數據的實際時間戳估計每年 bar 數，
        # 替代 MT5Backtest 默認的 H1=6240。A 股日線/15min、加密日線等
        # 非 H1 週期不再被按 H1 年化（否則 Sharpe/年化收益被放大數倍）。
        _dm_raw = getattr(self.data_manager, "raw_dict", None) or {}
        _dm_time = _dm_raw.get("time", None)
        if _dm_time is not None:
            try:
                _ppy = estimate_periods_per_year(_dm_time)
                if _ppy != self.bt.periods_per_year:
                    if verbose_header:
                        print(f"   年化因子: {_ppy} bar/年（按數據週期自動估計）")
                    self.bt.periods_per_year = _ppy
            except Exception:
                pass  # 估計失敗則保留默認 6240

        bs      = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new   = bs - n_elite

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[訓練] 起始步 {start_step} 已達目標步 {end_step}，無需繼續訓練。")
            return

        # 非交互/重定向輸出時關閉 tqdm 進度條，避免進度條洗版把自訂日誌淹掉。
        # tqdm.write 仍然可用，詳細 step 日誌會繼續輸出。
        pbar               = tqdm(range(start_step, end_step),
                                  total=end_step,
                                  initial=start_step,
                                  disable=not sys.stderr.isatty(),
                                  leave=False,
                                  mininterval=5.0)
        low_entropy_streak = 0
        prev_init_dist     = None  # 用於計算相鄰步分布差異 KL

        for step in pbar:
            # ── Part A: Sample n_new new formulas ────────────────────
            inp_new = torch.zeros((n_new, 1), dtype=torch.long,
                                  device=ModelConfig.DEVICE)
            lp_new, tok_new, ent_new = [], [], []
            sd_new = [0] * n_new
            prev_tokens_new: list[int | None] = [None] * n_new
            infected_chain_new: list[int] = [0] * n_new

            for si in range(ModelConfig.MAX_FORMULA_LEN):
                lg, _, _ = self.model(inp_new)
                lg = self.sampler.apply_mask_to_logits(lg, sd_new, si,
                                                       ModelConfig.MAX_FORMULA_LEN,
                                                       prev_tokens=prev_tokens_new,
                                                       infected_chain_lens=infected_chain_new)
                d  = Categorical(logits=lg)
                a  = d.sample()
                lp_new.append(d.log_prob(a))
                tok_new.append(a)
                ent_new.append(d.entropy())
                inp_new = torch.cat([inp_new, a.unsqueeze(1)], dim=1)
                for b in range(n_new):
                    sd_new[b] += self.sampler.delta[a[b].item()]
                    prev_tokens_new[b] = a[b].item()
                    infected_chain_new[b] = self.sampler.update_infection(
                        a[b].item(), infected_chain_new[b])

            seqs_new = torch.stack(tok_new, dim=1)


            # ── Part B: Elite Replay ─────────────────────────────────
            elite_formulas: list[list[int]] = []
            if self._elite_pool and n_elite > 0:
                ps = []
                pt = []
                weights = []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc)
                    pt.append(toks)
                    weights.append(decay)
                ps_min  = min(ps)
                ps_max  = max(ps)
                # 軟溫度採樣：避免最高分公式壟斷
                # 先歸一到 [0,1]，再除以溫度 T=0.5 後做 softmax
                # T<1 → 高分公式仍被偏好，但不再獨占
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp)) for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e   = random.choices(range(len(self._elite_pool)),
                                         weights=probs, k=n_elite)
                elite_formulas = [pt[i] for i in idx_e]

                # 詳細日誌：Elite Replay 衰減狀態（每 100 步列印一次）
                if step % 100 == 0:
                    avg_decay = sum(weights) / len(weights)
                    max_age = max(max(0, step - birth) for _, _, _, birth in self._elite_pool)
                    age_list = sorted([max(0, step - birth) for _, _, _, birth in self._elite_pool])
                    tqdm.write(
                        f"[精英重播 @ 第{step}步] 池大小={len(self._elite_pool)} "
                        f"平均衰減={avg_decay:.3f} 最大齡期={max_age} 齡期列表={age_list} "
                        f"抽樣分數=[{', '.join(f'{ps[i]:.3f}' for i in idx_e[:3])}...]"
                    )
            else:
                elite_formulas = seqs_new[:n_elite].tolist()

            lp_elite, ent_elite = [], []
            if elite_formulas:
                ne     = len(elite_formulas)
                inp_e  = torch.zeros((ne, 1), dtype=torch.long,
                                     device=ModelConfig.DEVICE)
                sd_e   = [0] * ne
                prev_tokens_elite: list[int | None] = [None] * ne
                infected_chain_elite: list[int] = [0] * ne
                tok_e_t = torch.tensor(elite_formulas, dtype=torch.long,
                                       device=ModelConfig.DEVICE)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    lg_e, _, _ = self.model(inp_e)
                    lg_e = self.sampler.apply_mask_to_logits(
                        lg_e, sd_e, si, ModelConfig.MAX_FORMULA_LEN,
                        prev_tokens=prev_tokens_elite,
                        infected_chain_lens=infected_chain_elite
                    )
                    d_e  = Categorical(logits=lg_e)
                    tk   = tok_e_t[:, si]
                    lp_elite.append(d_e.log_prob(tk))
                    ent_elite.append(d_e.entropy())
                    inp_e = torch.cat([inp_e, tk.unsqueeze(1)], dim=1)
                    for b in range(ne):
                        sd_e[b] += self.sampler.delta[tk[b].item()]
                        prev_tokens_elite[b] = tk[b].item()
                        infected_chain_elite[b] = self.sampler.update_infection(
                        tk[b].item(), infected_chain_elite[b])


            # ── Part C: Evaluate all formulas ────────────────────────
            all_fmls = seqs_new.tolist() + elite_formulas
            tot      = len(all_fmls)
            rewards    = torch.zeros(tot, device=ModelConfig.DEVICE)
            val_scores = torch.zeros(tot, device=ModelConfig.DEVICE)

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf');  step_best_f = None
            bic, bis, bsor = [], [], []

            for i, fml in enumerate(all_fmls):
                with torch.no_grad():
                    res = self.vm.execute(fml, feat)
                if res is None:
                    rewards[i] = val_scores[i] = -5.0
                    none_cnt += 1;  continue
                if res.std() < 1e-4:
                    rewards[i] = val_scores[i] = -2.0
                    const_cnt += 1;  continue
                ok_cnt += 1

                with torch.no_grad():
                    if use_wf:
                        fold_tr, fold_vl, fold_ic = [], [], []
                        for fold in folds:
                            # res[:, train_start:train_end] 是在已無洩露的因子上切片，正確
                            tr_sc, vl_sc = self.bt.evaluate_fold(
                                res, t_ret,
                                fold["train_start"], fold["train_end"],
                                fold["val_start"],   fold["val_end"],
                            )
                            ic_m, _ = AlphaEngine._compute_ic(
                                res[:, fold["train_start"]:fold["train_end"]],
                                t_ret[:, fold["train_start"]:fold["train_end"]],
                            )
                            tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                            fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                            fold_vl.append(vl_sc)
                            fold_ic.append(ic_m.item())
                        train_score = torch.stack(fold_tr).mean()
                        val_score   = torch.stack(fold_vl).mean()
                        ic_i        = sum(fold_ic) / len(fold_ic)
                    else:
                        train_score, _ = self.bt.evaluate(res, {}, t_ret)
                        ic_m0, _  = AlphaEngine._compute_ic(res, t_ret)
                        train_score = AlphaEngine._apply_ic_gate(
                            ModelConfig.REWARD_ALPHA * train_score, ic_m0
                        )
                        val_score = train_score
                        ic_i      = ic_m0.item()
                    ic_full, ic_stab_full = AlphaEngine._compute_ic(res, t_ret)

                rewards[i]    = train_score
                val_scores[i] = val_score
                bic.append(ic_full.item());  bis.append(ic_stab_full.item())
                bsor.append(val_score.item())

                # 重複懲罰和相關性懲罰同時施加到 rewards 和 val_scores
                # 保證 best_score / elite_pool 選優時已含所有懲罰
                rp = _repetition_penalty(fml)
                if rp > 0:
                    rewards[i]    -= rp
                    val_scores[i] -= rp
                rewards[i]    = self._apply_corr_penalty(rewards[i], res)
                val_scores[i] = self._apply_corr_penalty(val_scores[i], res)

                # 用含懲罰的 val_scores[i] 選全局最優
                final_val = val_scores[i].item()
                if final_val > step_max_val:
                    step_max_val = final_val;  step_best_f = fml

                if final_val > self.best_score:
                    # OOS 泛化門控：val_score / train_score < 0.5 說明過擬合
                    train_val = rewards[i].item()
                    if train_val > 0.5 and final_val < train_val * 0.5:
                        tqdm.write(
                            f"[過擬合跳過 @ 第{step}步] 驗證={final_val:.3f} "
                            f"訓練={train_val:.3f} 比值={final_val/train_val:.2f} | 樣本外表現過差"
                        )
                        pass
                    else:
                        # P3：冠軍選擇穩健性校驗——連續倉位均值 < 5% 的極稀疏公式拒絕登頂
                        pos_check = compute_target_positions_stateless(res)
                        exposure = pos_check.abs().mean().item()  # 連續倉位：均值
                        if exposure < 0.05:
                            # 極稀疏：參與梯度更新但不登頂，但記錄日誌方便排查
                            tqdm.write(
                                f"[稀疏跳過 @ 第{step}步] 驗證={final_val:.3f} "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | 倉位過稀疏，不更新最優"
                            )
                            pass
                        else:
                            old_best = self.best_score
                            self.best_score   = final_val
                            self.best_formula = fml
                            self._best_snapshot = copy.deepcopy(self.model.state_dict())
                            self._best_update_step = step
                            self._stagnation_steps = 0
                            self._update_factor_pool(final_val, res)
                            # 即時保存：任何時刻進程退出都有最新最優公式（防終端回收丟策略）
                            self._save_strategy_live()
                            tqdm.write(
                                f"[!] 新最優 @ 第{step}步: 驗證={final_val:.3f} "
                                f"(原 {old_best:.3f}，+{final_val-old_best:.3f}) "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | "
                                f"{fml}\n    {self._decode_formula(fml)}"
                            )
                self._update_elite_pool(final_val, fml, step)


            # ── Part D: REINFORCE gradient update ────────────────────
            # Fix 3: EMA baseline 替代 batch mean，避免全負 batch 的相對優選問題
            batch_mean = rewards.mean().item()
            batch_std  = rewards.std().clamp(min=0.1)
            if ModelConfig.REWARD_EMA_BASELINE and self._reward_ema_step >= ModelConfig.REWARD_EMA_WARMUP:
                baseline = self._reward_ema
                adv = (rewards - baseline) / (batch_std + 1e-5)
            else:
                adv = (rewards - batch_mean) / (batch_std + 1e-5)
            # 更新 EMA
            if self._reward_ema is None:
                self._reward_ema = batch_mean
            else:
                self._reward_ema = ModelConfig.REWARD_EMA_DECAY * self._reward_ema + (1.0 - ModelConfig.REWARD_EMA_DECAY) * batch_mean
            self._reward_ema_step += 1
            adv_new   = adv[:n_new]
            adv_elite = adv[n_new:]

            policy_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            for ti in range(len(lp_new)):
                policy_loss += (-lp_new[ti] * adv_new).mean()
            if lp_elite and adv_elite.shape[0] > 0:
                for ti in range(len(lp_elite)):
                    lpe = lp_elite[ti]
                    if lpe.shape[0] == adv_elite.shape[0]:
                        policy_loss += (-lpe * adv_elite
                                        * ModelConfig.ELITE_REWARD_SCALE).mean()

            if ent_new:
                mean_ent_new = torch.stack(ent_new).mean()
            else:
                mean_ent_new = torch.zeros(1, device=ModelConfig.DEVICE)
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (
                    mean_ent_new * n_new + mean_ent_elite * n_elite
                ) / (n_new + n_elite)
            else:
                mean_ent = mean_ent_new
            ent_val   = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER
            )
            # Fix 1: 熵下限懲罰——當 H < threshold 時加入固定懲罰，確保探索壓力不歸零
            ent_floor_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            if ModelConfig.ENTROPY_FLOOR and ent_val < ModelConfig.ENTROPY_FLOOR_THRESH:
                floor_gap = ModelConfig.ENTROPY_FLOOR_THRESH - ent_val
                ent_floor_loss = ModelConfig.ENTROPY_FLOOR_LAMBDA * torch.tensor(
                    floor_gap, device=ModelConfig.DEVICE, dtype=mean_ent.dtype
                )
            loss = policy_loss - ent_coeff * mean_ent + ent_floor_loss

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()

            # ── Part D2: 分布細化指標 ────────────────────────────────
            dst = self._distribution_stats(prev_init_dist)
            prev_init_dist = dst['dist']
            with torch.no_grad():
                uniq_tokens = seqs_new.unique().numel()
                uniq_fmls   = torch.unique(seqs_new, dim=0).shape[0]
                fml_div     = uniq_fmls / max(1, n_new)

            # ── Part E: Logging & history & checkpoint ───────────────
            avg_rew = rewards.mean().item()
            avg_val = val_scores.mean().item()
            bim  = sum(bic)  / len(bic)  if bic  else 0.0
            bis_ = sum(bis)  / len(bis)  if bis  else 0.0
            bsor_= sum(bsor) / len(bsor) if bsor else 0.0

            self._stagnation_steps = step - self._best_update_step
            tqdm.write(
                f"[{step+1}/{end_step}] "
                f"新公式={n_new} 精英={n_elite} | "
                f"有效={ok_cnt} 無效={none_cnt} 常數={const_cnt} | "
                f"獎勵={avg_rew:.3f} 驗證={avg_val:.3f} | "
                f"IC={bim:.4f} | 熵={ent_val:.3f}(係數={ent_coeff:.3f}) | "
                f"最優={self.best_score:.3f} 停滯={self._stagnation_steps} "
                f"精英池={len(self._elite_pool)} 重啟={self._restart_count}"
            )
            tqdm.write(
                f"   分布: 初始熵={dst['entropy']:.3f} KL均勻={dst['kl_uniform']:.3f} "
                f"KL上步={dst['kl_prev']:.4f} 最高機率={dst['top1_prob']:.3f} "
                f"前五機率={dst['top5_prob']:.3f} 有效詞彙={dst['eff_vocab']:.2f} "
                f"標準差={dst['prob_std']:.4f} | "
                f"本批: 唯一符號={uniq_tokens}/{FORMULA_VOCAB.size} "
                f"唯一公式={uniq_fmls}/{n_new} 多樣性={fml_div:.2f}"
            )
            pbar.set_postfix({
                '驗證': f"{avg_val:.3f}", '最優': f"{self.best_score:.3f}",
                '熵':   f"{ent_val:.2f}", 'IC':   f"{bim:.4f}",
                '停滯': f"{self._stagnation_steps}",
                '初始熵':  f"{dst['entropy']:.2f}",
                'KL上步': f"{dst['kl_prev']:.3f}",
            })

            if self.use_lord and step % 10 == 0:
                sr = self.rank_monitor.compute()
                self.training_history['stable_rank'].append(sr)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('ic_stability', []).append(bis_)
            self.training_history.setdefault('sortino', []).append(bsor_)
            self.training_history.setdefault('elite_pool_size', []).append(
                len(self._elite_pool))
            self.training_history.setdefault('init_entropy', []).append(dst['entropy'])
            self.training_history.setdefault('kl_uniform', []).append(dst['kl_uniform'])
            self.training_history.setdefault('kl_prev', []).append(dst['kl_prev'])
            self.training_history.setdefault('top1_prob', []).append(dst['top1_prob'])
            self.training_history.setdefault('eff_vocab', []).append(dst['eff_vocab'])
            self.training_history.setdefault('batch_uniq_tokens', []).append(uniq_tokens)
            self.training_history.setdefault('batch_uniq_fmls', []).append(uniq_fmls)
            self.training_history.setdefault('batch_fml_div', []).append(fml_div)

            if self.best_formula is not None:
                from .vocab import VOCAB_VERSION
                strategy_data = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": self.target_symbol,
                    "formula": self.best_formula,
                    "best_score": self.best_score,
                }
                save_path = _strategy_file_for_symbol(self.target_symbol)
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as fp:
                    json.dump(strategy_data, fp, indent=2)

            self._save_training_history_live()

            if (step + 1) % 20 == 0 or (step + 1) == end_step:
                ckpt = self.save_checkpoint(step + 1)
                tqdm.write(f"[檢查點] → {ckpt} (最優={self.best_score:.3f})")

            # ── Part F: Migration hook（多島訓練時交換精英）────────────
            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                tqdm.write(f"[遷移鉤子 @ 第{step+1}步] 調用已註冊鉤子")
                migration_hook(self, step + 1)

            # ── Part G: Entropy collapse detection & restart ─────────
            if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
                low_entropy_streak += 1
            else:
                low_entropy_streak  = 0

            if low_entropy_streak >= ModelConfig.ENTROPY_COLLAPSE_STEPS:
                # ── 自適應噪聲：根據 stagnation 調整 ─────────────────────
                self._stagnation_steps = step - self._best_update_step
                stagnation_ratio = self._stagnation_steps / max(1, ModelConfig.STAGNATION_WINDOW)
                base_noise = ModelConfig.RESTART_NOISE
                if ModelConfig.ADAPTIVE_NOISE:
                    raw_noise = base_noise + ModelConfig.NOISE_BOOST_FACTOR * 0.1 * min(stagnation_ratio, 3.0)
                    noise = max(ModelConfig.NOISE_MIN, min(ModelConfig.NOISE_MAX, raw_noise))
                else:
                    noise = base_noise

                max_r = ModelConfig.MAX_RESTARTS
                if self._restart_count < max_r:
                    self._restart_count  += 1
                    low_entropy_streak    = 0

                    # Fix 2: 每 N 次重啟做一次完全隨機初始化，逃離 best_snapshot 吸引子
                    # 深度坍塌 (H < 0.3) 時強制 full reset，不給 best_snapshot 恢復的機會
                    do_full_reset = (
                        self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                        or ent_val < 0.3
                    )

                    if do_full_reset:
                        # 完全重新初始化模型參數
                        for layer in self.model.modules():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        tqdm.write(
                            f"[重啟 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=完全重設（脫離最優快照吸引子） "
                            f"停滯={self._stagnation_steps} "
                            f"熵={ent_val:.3f}"
                        )
                    elif self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                        with torch.no_grad():
                            if ModelConfig.PARTIAL_RESET:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if any(k in nm for k in ModelConfig.PARTIAL_RESET_LAYERS):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                            else:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if 'ffn' in nm or 'attention' in nm or nm.startswith('blocks'):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                        tqdm.write(
                            f"[重啟 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式={'部分層' if ModelConfig.PARTIAL_RESET else 'FFN/注意力'} "
                            f"噪聲={noise:.4f}(基準={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滯={self._stagnation_steps} "
                            f"熵={ent_val:.3f} "
                            f"擾動層數={len(perturbed_layers)}"
                        )
                    else:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                p.add_(torch.randn_like(p) * noise)
                        tqdm.write(
                            f"[重啟 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=全參數 "
                            f"噪聲={noise:.4f}(基準={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滯={self._stagnation_steps} "
                            f"熵={ent_val:.3f} | 無最優快照"
                        )
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                else:
                    # 訓練時間不敏感：超過重啟上限後不再 Early Stop 終止，
                    # 改為「全參數強擾動 + 重設流計數」繼續探索，直到跑滿 TRAIN_STEPS。
                    # 從 best_snapshot 恢復（若有）以保住已發現的最優結構，再加大擾動。
                    low_entropy_streak = 0
                    hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * hard_noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(
                        f"[強重啟 @ 第{step}步] 已達最大重啟次數={max_r} "
                        f"熵={ent_val:.3f} 強噪聲={hard_noise:.4f} "
                        f"繼續訓練，不提前停止"
                    )

        # ── End of training ──────────────────────────────────────────
        # 僅當跑滿最終步時才保存最終 strategy 和歷史
        if end_step == ModelConfig.TRAIN_STEPS:
            if self.best_formula is not None:
                from .vocab import VOCAB_VERSION
                strategy_data = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": self.target_symbol,
                    "formula": self.best_formula,
                    "best_score": self.best_score,
                }
                save_path = _strategy_file_for_symbol(self.target_symbol)
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as fp:
                    json.dump(strategy_data, fp, indent=2)

            sym_tag = f"[{self.target_symbol}] " if self.target_symbol else ""
            self.training_history.pop('_low_entropy_streak', None)
            hist_path = (
                f"training_history_{self.target_symbol}.json"
                if self.target_symbol else "training_history.json"
            )
            with open(hist_path, "w") as fp:
                json.dump(self.training_history, fp)

            print(f"\n[完成] {sym_tag}訓練結束！")
            print(f"  最優驗證分數 : {self.best_score:.4f}")
            print(f"  最優公式令牌 : {self.best_formula}")
            print(f"  可讀公式     : {self._decode_formula(self.best_formula)}")
            print(f"  精英池大小   : {len(self._elite_pool)}")
            print(f"  精英衰減     : 啟用={ModelConfig.ELITE_DECAY}，半衰期={ModelConfig.ELITE_DECAY_HALF_LIFE}")
            print(f"  自適應噪聲   : 啟用={ModelConfig.ADAPTIVE_NOISE}，範圍=[{ModelConfig.NOISE_MIN}, {ModelConfig.NOISE_MAX}]")
            print(f"  部分層重設   : 啟用={ModelConfig.PARTIAL_RESET}，層={ModelConfig.PARTIAL_RESET_LAYERS}")
            print(f"  重啟次數     : {self._restart_count}")
            print(f"  策略已保存   : {save_path}")


    # ── 即時保存最優公式（防進程意外退出遺失）────────────────────────────────
    def _save_training_history_live(self) -> None:
        """週期性寫入訓練曲線 JSON，供 Web UI 即時展示。"""
        if not self.target_symbol:
            return
        try:
            hist_path = f"training_history_{self.target_symbol}.json"
            payload = {
                k: v for k, v in self.training_history.items()
                if k != "_low_entropy_streak"
            }
            with open(hist_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp)
        except Exception:
            pass

    def _save_strategy_live(self) -> None:
        """每次 best_formula 更新時立即保存 strategy json。
        即使訓練中途進程被殺（OOM/終端回收/Ctrl+C），也能保留最新最優公式。
        """
        if self.best_formula is None:
            return
        try:
            from .vocab import VOCAB_VERSION
            save_path = _strategy_file_for_symbol(self.target_symbol)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            existing: dict = {}
            p = pathlib.Path(save_path)
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        existing = raw
                except Exception:
                    existing = {}

            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "symbol": self.target_symbol,
                "formula": self.best_formula,
                "best_score": self.best_score,
                "formula_decoded": self._decode_formula(self.best_formula),
            }
            # 保留訓練數據路徑等元數據，避免 live 保存把 data_file 沖掉
            for key in ("timeframe", "data_file", "mode", "train_steps"):
                val = getattr(self, key, None)
                if val is None:
                    val = existing.get(key)
                if val is not None:
                    strategy_data[key] = val
            if not strategy_data.get("data_file") and self.target_symbol:
                data_file, tf = _fallback_data_file_for_symbol(self.target_symbol)
                if data_file:
                    strategy_data["data_file"] = data_file
                if tf and not strategy_data.get("timeframe"):
                    strategy_data["timeframe"] = tf
                if data_file and not strategy_data.get("mode"):
                    strategy_data["mode"] = "parquet_file"

            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
            path = str(_CHECKPOINT_DIR / f"ckpt{sym_tag}_step_{step:04d}.pt")
        ckpt = {
            "step":                 step,
            "vocab_version":        VOCAB_VERSION,   # task 12.2: 版本校驗所需
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "elite_pool":           self._elite_pool,
            "elite_counter":        self._elite_counter,
            "restart_count":        self._restart_count,
            "training_history":     {
                k: v for k, v in self.training_history.items()
                if k != '_low_entropy_streak'
            },
        }
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)

        # ── Task 12.2：版本校驗（R3.7）──────────────────────────────────────
        # 從 checkpoint 讀取 vocab_version；若欄位缺失（舊版 checkpoint），視為
        # 版本不匹配並拋錯——拒絕載入、不消費任何 token。
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError(
                f"checkpoint '{path}' 不含 vocab_version 欄位（舊版產物），"
                f"當前詞表版本 {FORMULA_VOCAB.version!r}；需重新訓練後載入"
            )
        # verify() 版本不匹配時拋 VocabVersionMismatchError，拒絕載入
        FORMULA_VOCAB.verify(artifact_version)
        # ── 版本校驗通過，繼續載入 ────────────────────────────────────────

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_score          = ckpt.get("best_score",  -float('inf'))
        self.best_formula        = ckpt.get("best_formula", None)
        self._best_snapshot      = ckpt.get("best_snapshot", None)
        self.factor_pool         = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        self._elite_pool         = ckpt.get("elite_pool", [])
        self._elite_counter      = ckpt.get("elite_counter", 0)
        self._restart_count      = ckpt.get("restart_count", 0)
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v

        # 清理 elite pool 中的重複條目（保留各公式的最高分版本）
        self._elite_pool = self._dedup_elite_pool(self._elite_pool)

        completed = ckpt.get("step", 0)
        tqdm.write(f"[檢查點] 已從 {path} 恢復。"
                   f" 當前步={completed}  最優={self.best_score:.4f}"
                   f"  精英池={len(self._elite_pool)}（去重後）")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "無"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)
