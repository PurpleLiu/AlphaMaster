"""
data_pipeline/data_manager.py — 多品種數據管理模組

負責從 MT5DataFetcher 載入多品種 OHLCV 數據，執行時間軸對齊，
構建 raw_dict、target_ret 張量，並委託 MT5FeatureEngineer 生成特徵張量。
數據快取在記憶體中，通過 reload() 刷新。
"""

from __future__ import annotations

import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.fetcher import MT5DataFetcher


class MT5DataManager:
    """多品種數據管理器。

    用法：
        with MT5DataFetcher() as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            tensor = mgr.feat_tensor   # [N, 6, T]
            ret    = mgr.target_ret    # [N, T]

    Args:
        fetcher: 已連接的 MT5DataFetcher 實例（可通過上下文管理器或
                 手動調用 connect() 確保已連接）。
    """

    def __init__(self, fetcher: MT5DataFetcher) -> None:
        self._fetcher = fetcher

        # 快取狀態
        self._symbols: list[str] = []          # 有效品種列表（>= MIN_BARS）
        self._raw_dict: dict | None = None     # {field: Tensor[N, T]}
        self._target_ret: torch.Tensor | None = None  # [N, T]

    # ──────────────────────────────────────────────────────────────────────
    # 公開方法
    # ──────────────────────────────────────────────────────────────────────

    def load(self, symbols: list[str] | None = None) -> None:
        """載入指定品種（默認 Config.SYMBOLS）的 OHLCV 數據到記憶體。

        - 遍歷 Config.SYMBOLS，調用 fetcher.fetch() 獲取每個品種數據。
        - 排除 bars < Config.MIN_BARS 的品種並記錄 WARNING。
        - 對剩餘品種做時間軸對齊（時間戳併集 + forward-fill）。
        - 構建 raw_dict 和 target_ret 並快取。

        Raises:
            ValueError: 所有品種均不滿足 MIN_BARS 要求時拋出。
        """
        symbol_list = list(symbols) if symbols is not None else list(Config.SYMBOLS)
        logger.info(
            f"Loading data for {len(symbol_list)} symbols: {symbol_list}"
        )

        # ── 步驟 1：拉取原始數據 ────────────────────────────────────────
        raw_dfs: dict[str, pd.DataFrame] = {}
        for symbol in symbol_list:
            df = self._fetcher.fetch(symbol, Config.TIMEFRAME, Config.BARS_COUNT)
            if len(df) < Config.MIN_BARS:
                logger.warning(
                    f"Symbol '{symbol}' has only {len(df)} bars "
                    f"(< MIN_BARS={Config.MIN_BARS}). Excluding."
                )
                continue
            raw_dfs[symbol] = df

        if not raw_dfs:
            raise ValueError(
                "No valid symbols loaded: all symbols have fewer than "
                f"{Config.MIN_BARS} bars."
            )

        self._symbols = list(raw_dfs.keys())
        logger.info(
            f"Valid symbols ({len(self._symbols)}): {self._symbols}"
        )

        # ── 步驟 2：時間軸對齊 ────────────────────────────────────────
        aligned = self._align_timelines(raw_dfs)

        # ── 步驟 3：構建 raw_dict ─────────────────────────────────────
        self._raw_dict = self._build_raw_dict(aligned)

        # ── 步驟 4：計算 target_ret ───────────────────────────────────
        self._target_ret = self._compute_target_ret(self._raw_dict["open"])

        logger.info(
            f"Data loaded. raw_dict shape: N={len(self._symbols)}, "
            f"T={self._raw_dict['open'].shape[1]}"
        )

    def reload(self) -> None:
        """刷新快取：清空檔前快取並重新從 MT5 載入。"""
        logger.info("Reloading data from MT5...")
        self._symbols = []
        self._raw_dict = None
        self._target_ret = None
        self.load()

    # ──────────────────────────────────────────────────────────────────────
    # 屬性
    # ──────────────────────────────────────────────────────────────────────

    @property
    def raw_dict(self) -> dict:
        """返回 OHLCV 原始張量字典。

        Returns:
            dict，鍵為 "open"、"high"、"low"、"close"、"volume"，
            每個值為形狀 [N, T] 的 torch.Tensor（float32）。

        Raises:
            RuntimeError: 未調用 load() 時訪問此屬性。
        """
        self._ensure_loaded()
        return self._raw_dict  # type: ignore[return-value]

    @property
    def feat_tensor(self) -> torch.Tensor:
        """返回特徵張量，形狀 [N, F, T]（F=6）。

        委託 MT5FeatureEngineer.compute_features(raw_dict) 計算。
        使用懶導入避免循環依賴；若 model_core.features 尚未創建，
        返回全零占位張量。

        Raises:
            RuntimeError: 未調用 load() 時訪問此屬性。
        """
        self._ensure_loaded()
        raw = self._raw_dict  # type: ignore[assignment]

        try:
            from model_core.features import MT5FeatureEngineer  # lazy import
            # 因果安全：MT5FeatureEngineer._robust_norm 已改為滾動因果實現，
            # 全量序列傳入不引入 look-ahead 洩露
            return MT5FeatureEngineer.compute_features(raw)
        except ImportError:
            # model_core/features.py 尚未在任務 5.1 中創建
            logger.warning(
                "model_core.features not found (task 5.1 not yet implemented). "
                "Returning zero placeholder feat_tensor."
            )
            n = len(self._symbols)
            t = raw["open"].shape[1]
            f = Config.INPUT_DIM
            return torch.zeros(n, f, t, dtype=torch.float32)

    @property
    def target_ret(self) -> torch.Tensor:
        """返回目標收益率張量，形狀 [N, T]。

        target_ret[n, t] = log(open[n, t+2] / open[n, t+1])
        最後兩個時間步設為 0（邊界）。

        Raises:
            RuntimeError: 未調用 load() 時訪問此屬性。
        """
        self._ensure_loaded()
        return self._target_ret  # type: ignore[return-value]

    @property
    def bar_time(self) -> torch.Tensor:
        """返回最新已收盤 K 線的時間戳張量，形狀 [N]（Unix 秒，int64）。

        用於實盤 runner 檢測新 K 線收盤：
            if (bar_time != last_bar_time).any(): 觸發調倉

        當 raw_dict 中沒有 "time" 欄位時（老版本 data_manager）返回全零張量。
        """
        self._ensure_loaded()
        raw = self._raw_dict
        if "time" in raw:
            # raw_dict["time"] 形狀 [N, T]，取最後一列
            return raw["time"][:, -1].long()
        n = len(self._symbols)
        return torch.zeros(n, dtype=torch.int64)

    @property
    def symbols(self) -> list[str]:
        """返回有效品種列表（bars >= MIN_BARS 的品種）。"""
        return list(self._symbols)

    # ──────────────────────────────────────────────────────────────────────
    # 內部輔助方法
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """確保數據已載入，否則拋出 RuntimeError。"""
        if self._raw_dict is None:
            raise RuntimeError(
                "Data not loaded. Call MT5DataManager.load() first."
            )

    def _align_timelines(
        self, raw_dfs: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """將多品種 DataFrame 對齊到統一時間軸（時間戳交集）。

        使用交集而非併集：只保留所有品種都有真實報價的時間戳，
        徹底消除因休市 forward-fill 導致的"重複K線"問題。

        Args:
            raw_dfs: 品種名 → DataFrame（含 time 列，以及 OHLCV 列）的字典。

        Returns:
            品種名 → 對齊後 DataFrame（以 time 為索引，含 open/high/low/close/volume 列）。
            所有品種行數完全相同，無任何 NaN（交集保證每個時間戳各品種均有真實數據）。
        """
        fields = ["open", "high", "low", "close", "volume"]

        # 為每個品種建立以 time 為索引的 DataFrame
        indexed: dict[str, pd.DataFrame] = {}
        for symbol, df in raw_dfs.items():
            volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
            sub = df[["time", "open", "high", "low", "close", volume_col]].copy()
            sub = sub.rename(columns={volume_col: "volume"})
            sub = sub.set_index("time")
            # 去掉同一時間戳重複的行（取最後一條）
            sub = sub[~sub.index.duplicated(keep="last")]
            indexed[symbol] = sub

        # 構建時間戳交集索引：只保留所有品種都有報價的 bar
        inter_index: pd.Index = next(iter(indexed.values())).index
        for sub in indexed.values():
            inter_index = inter_index.intersection(sub.index)
        inter_index = inter_index.sort_values()

        if len(inter_index) < Config.MIN_BARS:
            # 交集太小時降級回併集+ffill，並記錄警告
            logger.warning(
                f"Intersection timeline has only {len(inter_index)} bars "
                f"(< MIN_BARS={Config.MIN_BARS}). Falling back to union+ffill."
            )
            union_index: pd.Index = pd.Index([], dtype="int64")
            for sub in indexed.values():
                union_index = union_index.union(sub.index)
            union_index = union_index.sort_values()
            aligned: dict[str, pd.DataFrame] = {}
            for symbol, sub in indexed.items():
                reindexed = sub.reindex(union_index)
                reindexed = reindexed.ffill().bfill()
                aligned[symbol] = reindexed[fields]
            return aligned

        logger.info(
            f"Intersection timeline: {len(inter_index)} bars "
            f"(from union of {sum(len(s) for s in indexed.values())} total)"
        )

        # 用交集索引直接切片，無需 ffill
        aligned = {}
        for symbol, sub in indexed.items():
            aligned[symbol] = sub.reindex(inter_index)[fields]

        return aligned

    def _build_raw_dict(
        self, aligned: dict[str, pd.DataFrame]
    ) -> dict[str, torch.Tensor]:
        """將對齊後的 DataFrames 轉換為 {field: Tensor[N, T]} 格式。

        Args:
            aligned: 品種名 → 對齊 DataFrame 的字典。

        Returns:
            字典，鍵為 "open"、"high"、"low"、"close"、"volume"，
            值為 torch.float32 張量，形狀 [N, T]。
        """
        fields = ["open", "high", "low", "close", "volume"]
        n = len(self._symbols)
        t = next(iter(aligned.values())).shape[0]

        raw_dict: dict[str, torch.Tensor] = {}
        for field in fields:
            rows = []
            for symbol in self._symbols:
                series = aligned[symbol][field].values
                rows.append(series)

            import numpy as np
            tensor = torch.tensor(np.array(rows), dtype=torch.float32)  # [N, T]
            raw_dict[field] = tensor

        # 加入時間戳欄位，供 bar_time 屬性和 K 線收盤檢測使用
        import numpy as np
        time_rows = []
        for symbol in self._symbols:
            # aligned 的 index 是時間戳（Unix 秒，int64）
            time_rows.append(aligned[symbol].index.values.astype("int64"))
        raw_dict["time"] = torch.tensor(np.array(time_rows), dtype=torch.int64)  # [N, T]

        assert raw_dict["open"].shape == (n, t)
        return raw_dict

    @staticmethod
    def _compute_target_ret(open_tensor: torch.Tensor) -> torch.Tensor:
        """計算目標收益率張量。

        target_ret[n, t] = log(open[n, t+2] / open[n, t+1])，對 t ∈ [0, T-3]
        最後兩個位置（t = T-2, T-1）設為 0（邊界）。

        Args:
            open_tensor: 形狀 [N, T] 的 open 價格張量（float32）。

        Returns:
            形狀 [N, T] 的 target_ret 張量（float32）。
        """
        n, t = open_tensor.shape
        target = torch.zeros(n, t, dtype=torch.float32)

        if t >= 3:
            # open[t+2] 對應索引 2..T-1，open[t+1] 對應索引 1..T-2
            numerator   = open_tensor[:, 2:]    # [N, T-2]
            denominator = open_tensor[:, 1:-1]  # [N, T-2]

            # 防止除以零（價格理應 > 0，但防禦性處理）
            safe_denom = denominator.clone()
            safe_denom[safe_denom == 0] = 1.0

            log_ret = torch.log(numerator / safe_denom)  # [N, T-2]
            target[:, :t - 2] = log_ret

        # 最後兩個時間步已在初始化時設為 0（torch.zeros）

        return target
