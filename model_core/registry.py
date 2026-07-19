"""
model_core/registry.py -- 聲明式註冊層（Registration_Interface, R10）

提供 Feature 與 Operator 的同構聲明式註冊機制：每個 Feature/Operator 表達為
單個聲明條目（`FeatureSpec` / `OperatorSpec`），通過 `Registry` 追加到有序列表
尾部。註冊時做同步校驗，遵循「先校驗、全部通過才追加」的原子性約定——任何一個
校驗失敗都不會改動註冊表，保證 Formula_Vocabulary 保持不變。

校驗與異常映射（對應 requirements R10.5–R10.8）：
  - 重複名                          -> DuplicateNameError   (R10.5)
  - 聲明 arity 與 transform 實際操作數不符 -> ArityMismatchError (R10.6)
  - 缺必填欄位                       -> MissingFieldError    (R10.7)
  - arity 非 0..10 整數              -> InvalidArityError    (R10.8)

name 約定：非空字串，長度 1..64（R10.1, R10.2）。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import torch

# arity 聲明區間（R10.1）：允許 0..10；進 VM 執行時另有 1..3 的更嚴約束（見 vocab/vm）。
_ARITY_MIN = 0
_ARITY_MAX = 10

# name 長度約束（R10.1, R10.2）
_NAME_MIN_LEN = 1
_NAME_MAX_LEN = 64


# ── 異常類型（對應 design「錯誤類型模型」）─────────────────────────────────

class RegistrationError(Exception):
    """註冊層錯誤基類。"""


class DuplicateNameError(RegistrationError):
    """註冊名與既有 token 衝突（R10.5）。"""


class ArityMismatchError(RegistrationError):
    """聲明 arity 與 transform 實際操作數不符（R10.6）。"""


class MissingFieldError(RegistrationError):
    """註冊條目缺必填欄位（R10.7）。"""


class InvalidArityError(RegistrationError):
    """arity 非 0..10 整數（R10.8）。"""


# ── 聲明條目（frozen dataclass）─────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSpec:
    """Feature 聲明條目。

    name:     1..64 字元、非空、唯一的字串標識。
    category: 類別標籤（trend/momentum/volatility/volume/reversal/channel/
              statistical/cross_sectional），用於報告分組與類別覆蓋校驗。
    compute:  計算函數，簽名 `(raw_dict: dict) -> Tensor[N, T]`。
    """
    name: str
    category: str
    compute: Callable[[dict], "torch.Tensor"]


@dataclass(frozen=True)
class OperatorSpec:
    """Operator 聲明條目。

    name:      1..64 字元、非空、唯一的字串標識。
    arity:     聲明區間 0..10 的整數；實際入 VM 執行時另要求 1..3。
    transform: 變換函數，簽名 `(*operands: Tensor[N, T]) -> Tensor[N, T]`。
    """
    name: str
    arity: int
    transform: Callable[..., "torch.Tensor"]


# ── 校驗輔助 ────────────────────────────────────────────────────────────

def _validate_name(name) -> None:
    """校驗 name 為 1..64 字元的非空字串（R10.1, R10.2）。

    缺失（None/空/純空白）視為缺欄位 -> MissingFieldError。
    """
    if name is None or (isinstance(name, str) and name.strip() == ""):
        raise MissingFieldError("缺少必填欄位: name")
    if not isinstance(name, str):
        raise MissingFieldError(
            f"欄位 name 必須為字串，實際類型為 {type(name).__name__}"
        )
    if not (_NAME_MIN_LEN <= len(name) <= _NAME_MAX_LEN):
        raise MissingFieldError(
            f"欄位 name 長度必須在 {_NAME_MIN_LEN}..{_NAME_MAX_LEN} 之間，"
            f"實際長度為 {len(name)}"
        )


def _observed_arity(transform: Callable) -> int | None:
    """觀測 transform 實際消費的操作數個數。

    統計必填的位置參數（無預設值、非 *args/**kwargs）。若簽名不可解析或含
    可變位置參數（*args），返回 None 表示「無法確定」，此時跳過 arity 匹配校驗。
    """
    try:
        sig = inspect.signature(transform)
    except (TypeError, ValueError):
        return None

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return None  # 可變位置參數，無法確定確切操作數

    count = 0
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if p.default is inspect.Parameter.empty:
                count += 1
    return count


# ── 註冊表 ──────────────────────────────────────────────────────────────

class Registry:
    """Feature / Operator 的有序聲明式註冊表。

    維護兩個有序列表（按註冊順序），並用集合保證 token 名稱跨 Feature/Operator
    全局唯一。註冊遵循「先校驗、全部通過才追加」的原子性約定。
    """

    def __init__(self) -> None:
        self._feature_specs: list[FeatureSpec] = []
        self._operator_specs: list[OperatorSpec] = []
        # 跨 Feature/Operator 的全局唯一名稱集合，用於 O(1) 重名檢測
        self._names: set[str] = set()

    # ── 只讀視圖 ────────────────────────────────────────────────────────

    @property
    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return tuple(self._feature_specs)

    @property
    def operator_specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(self._operator_specs)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._feature_specs)

    @property
    def operator_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._operator_specs)

    def __contains__(self, name: str) -> bool:
        return name in self._names

    # ── 註冊介面 ────────────────────────────────────────────────────────

    def register_feature(self, spec: FeatureSpec) -> None:
        """註冊單個 Feature（R10.2）。校驗全部通過後才追加，否則註冊表不變。"""
        # 1) 必填欄位校驗（R10.7）：name / category / compute
        self._require_fields(
            spec,
            {"name": "name", "category": "category", "compute": "compute"},
        )
        # 2) name 合法性（R10.2）
        _validate_name(spec.name)
        # 3) 重名校驗（R10.5）
        self._check_duplicate(spec.name)

        # 全部通過 -> 原子追加
        self._feature_specs.append(spec)
        self._names.add(spec.name)

    def register_operator(self, spec: OperatorSpec) -> None:
        """註冊單個 Operator（R10.1）。校驗全部通過後才追加，否則註冊表不變。"""
        # 1) 必填欄位校驗（R10.7）：name / arity / transform
        #    arity 為 0 是合法值，故用「是否為 None」判斷缺失而非真值判斷。
        self._require_fields(
            spec,
            {"name": "name", "arity": "arity", "transform": "transform"},
        )
        # 2) name 合法性（R10.1）
        _validate_name(spec.name)
        # 3) arity 類型與範圍（R10.8）：必須為 0..10 的整數（bool 不算整數）
        self._validate_arity(spec.arity)
        # 4) arity 與 transform 實際操作數一致性（R10.6）
        observed = _observed_arity(spec.transform)
        if observed is not None and observed != spec.arity:
            raise ArityMismatchError(
                f"運算元 '{spec.name}' 聲明 arity={spec.arity}，"
                f"但 transform 實際消費 {observed} 個操作數"
            )
        # 5) 重名校驗（R10.5）
        self._check_duplicate(spec.name)

        # 全部通過 -> 原子追加
        self._operator_specs.append(spec)
        self._names.add(spec.name)

    # ── 內部校驗 ────────────────────────────────────────────────────────

    @staticmethod
    def _require_fields(spec, fields: dict[str, str]) -> None:
        """校驗 spec 上每個必填欄位非 None（R10.7）。"""
        for attr, label in fields.items():
            if getattr(spec, attr, None) is None:
                raise MissingFieldError(f"缺少必填欄位: {label}")

    @staticmethod
    def _validate_arity(arity) -> None:
        """校驗 arity 為 0..10 的整數（R10.8）。bool 是 int 的子類，需顯式排除。"""
        if isinstance(arity, bool) or not isinstance(arity, int):
            raise InvalidArityError(
                f"arity 必須為 {_ARITY_MIN}..{_ARITY_MAX} 的整數，"
                f"實際類型為 {type(arity).__name__}"
            )
        if not (_ARITY_MIN <= arity <= _ARITY_MAX):
            raise InvalidArityError(
                f"arity 必須為 {_ARITY_MIN}..{_ARITY_MAX} 的整數，實際為 {arity}"
            )

    def _check_duplicate(self, name: str) -> None:
        """跨 Feature/Operator 全局重名檢測（R10.5）。"""
        if name in self._names:
            raise DuplicateNameError(f"註冊名衝突: '{name}' 已存在")
