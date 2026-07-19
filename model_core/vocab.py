"""
model_core/vocab.py -- Formula_Vocabulary 集成與確定性版本（R3）

本模組把 Formula_Vocabulary 從「手工維護的特徵名元組 + 手工版本字串」遷移為
由註冊層（`model_core.registry.Registry`）驅動、版本確定性派生的實現：

  - `feature_names` 來自 `features.FEATURE_REGISTRY`（有序）。
  - `operator_names` 來自 `ops.OPERATOR_REGISTRY`（有序）。
  - token id 分段：feature id ∈ [0, F-1]，operator id ∈ [F, F+O-1]，兩段嚴格
    不相交（`operator_offset == feature_count`，R3.3）。
  - 構建時用集合校驗 token 名稱全局唯一、無缺失/重複/多餘（R3.1、R3.2）。
  - `VOCAB_VERSION` 由有序 token 名稱列表確定性派生（R3.4、R3.5）：
        VOCAB_VERSION = "v" + sha256("\n".join(token_names)).hexdigest()[:12]
    相同有序列表 → 相同版本；任意組成/順序變化 → 不同版本。
  - `FORMULA_VOCAB.verify(artifact_version)`：版本不匹配拋
    `VocabVersionMismatchError`，拒絕且不消費任何 token（R3.7）。
  - `VOCAB_SCHEMA_TAG`：人類可讀的 schema 標籤，僅供日誌展示，不參與相容判定。

import 方向說明：`features.py` / `ops.py` 只依賴 `.registry`，本模組從二者讀取
註冊表視圖不構成循環依賴。下游 `vm.py` / `config.py` /
`alphagpt.py` / `engine.py` 對 `FEATURE_NAMES` / `FORMULA_VOCAB` / `VOCAB_VERSION`
的 import 保持相容。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .features import FEATURE_REGISTRY
from .ops import OPERATOR_REGISTRY

# 人類可讀 schema 標籤（僅供日誌/報告展示，不參與相容性判定，R3.5）
VOCAB_SCHEMA_TAG = "4.0-registry"


# ── 版本層異常（R3.7）───────────────────────────────────────────────────

class VocabVersionMismatchError(Exception):
    """載入產物版本 ≠ 當前派生 VOCAB_VERSION（R3.7）。

    由 `FORMULA_VOCAB.verify()` 在版本不匹配時拋出；調用方應拒絕載入且不消費
    任何 token。
    """


# ── 確定性版本派生（R3.4、R3.5）─────────────────────────────────────────

def compute_vocab_version(token_names: tuple[str, ...]) -> str:
    """由有序 token 名稱列表確定性派生緊湊版本標識。

    VOCAB_VERSION = "v" + sha256("\n".join(token_names)).hexdigest()[:12]

    性質：相同的有序列表 → 相同版本；任意組成或順序變化 → 不同版本。使用換行
    作為穩定分隔符號，避免名稱拼接歧義。
    """
    joined = "\n".join(token_names)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return "v" + digest[:12]


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        # feature id 段 [0, F-1]、operator id 段 [F, F+O-1] 嚴格不相交（R3.3）
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)

    @property
    def version(self) -> str:
        """當前詞表組成確定性派生的 VOCAB_VERSION（R3.4）。"""
        return compute_vocab_version(self.token_names)

    def verify(self, artifact_version: str) -> None:
        """校驗產物版本與當前派生版本一致（R3.7）。

        不匹配拋 `VocabVersionMismatchError`，調用方據此拒絕載入、不消費任何
        token。匹配則靜默返回。
        """
        current = self.version
        if artifact_version != current:
            raise VocabVersionMismatchError(
                f"詞表版本不匹配：產物版本 {artifact_version!r} != "
                f"當前派生版本 {current!r}；舊 checkpoint / best_strategy.json "
                f"需重新訓練/重建後載入"
            )


# ── 構建 FORMULA_VOCAB（由 registry 派生）與完整性校驗（R3.1、R3.2）──────

def _build_formula_vocab() -> FormulaVocab:
    """由 FEATURE_REGISTRY / OPERATOR_REGISTRY 構建詞表並做完整性校驗。

    校驗（用集合，R3.1、R3.2）：
      - feature / operator 名稱各自無重複；
      - feature 與 operator 名稱跨段全局唯一（無交集）；
      - size == F + O（無缺失/多餘）。
    任一校驗失敗即拋錯，不產出不一致的詞表。
    """
    feature_names = tuple(FEATURE_REGISTRY.feature_names)
    operator_names = tuple(OPERATOR_REGISTRY.operator_names)

    feat_set = set(feature_names)
    op_set = set(operator_names)

    # 段內唯一
    if len(feat_set) != len(feature_names):
        dup = sorted({n for n in feature_names if feature_names.count(n) > 1})
        raise ValueError(f"feature 名稱存在重複: {dup}")
    if len(op_set) != len(operator_names):
        dup = sorted({n for n in operator_names if operator_names.count(n) > 1})
        raise ValueError(f"operator 名稱存在重複: {dup}")

    # 跨段全局唯一（feature/operator 名稱不得衝突）
    overlap = feat_set & op_set
    if overlap:
        raise ValueError(f"feature 與 operator 名稱衝突: {sorted(overlap)}")

    vocab = FormulaVocab(feature_names=feature_names, operator_names=operator_names)

    # 計數一致性：size == F + O，無缺失/重複/多餘（R3.2）
    expected = len(feature_names) + len(operator_names)
    if vocab.size != expected:
        raise ValueError(
            f"詞表計數不一致: size={vocab.size} != F+O={expected}"
        )
    # 全局 token 名稱唯一（無缺失/重複/多餘）
    if len(set(vocab.token_names)) != vocab.size:
        raise ValueError("token 名稱存在重複或缺失，詞表完整性校驗失敗")

    return vocab


FORMULA_VOCAB = _build_formula_vocab()

# 由註冊表導出有序特徵名視圖（保持下游 import 相容）
FEATURE_NAMES = FORMULA_VOCAB.feature_names

# 詞表版本：由有序 token 名稱列表確定性派生（R3.4、R3.5）。
# 特徵/運算元的組成或順序變化都會改變本值，舊 checkpoint / best_strategy.json 將
# 因版本不匹配而被 verify() 拒絕載入。
VOCAB_VERSION = FORMULA_VOCAB.version
