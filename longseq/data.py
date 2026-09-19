"""读取一次性生成的长序列缓存，不再访问原始 CSV。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = [
    "mean_cell_voltage",
    "max_cell_voltage",
    "min_cell_voltage",
    "pack_current",
    "max_temperature",
    "min_temperature",
    "cell_voltage_spread",
]
CLASS_NAMES = ["normal", "low_capacity", "high_resistance", "self_discharge"]
CHEMISTRY_NAMES = ["LFP", "NCM"]
SPLIT_NAMES = ["train", "val", "test"]


def load_metadata(cache_dir: str | Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("format_version") != 1:
        raise ValueError(f"缓存未完成或版本不兼容: {path}")
    return metadata


class LongSequenceDataset(Dataset):
    """一个索引项是一段真实时间序列，不会跨 CSV 或车辆。"""

    def __init__(self, cache_dir: str | Path, split: str, chemistry: str | None = None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split 必须为 {SPLIT_NAMES}")
        if chemistry is not None and chemistry not in CHEMISTRY_NAMES:
            raise ValueError(f"chemistry 必须为 {CHEMISTRY_NAMES}")
        self.cache_dir = Path(cache_dir)
        self.metadata = load_metadata(self.cache_dir)
        self.window_length = int(self.metadata["window_length"])
        self.total_rows = int(self.metadata["total_rows"])
        arrays = np.load(self.cache_dir / "index.npz", allow_pickle=False)
        self.series_offset = arrays["series_offset"]
        self.series_length = arrays["series_length"]
        self.series_label = arrays["series_label"]
        self.series_chemistry = arrays["series_chemistry"]
        self.series_split = arrays["series_split"]
        self.series_vehicle = arrays["series_vehicle"]
        self.window_series = arrays["window_series"]
        self.window_start = arrays["window_start"]
        self.window_valid = arrays["window_valid"]
        selected = self.series_split[self.window_series] == SPLIT_NAMES.index(split)
        if chemistry is not None:
            selected &= self.series_chemistry[self.window_series] == CHEMISTRY_NAMES.index(chemistry)
        self.indices = np.flatnonzero(selected)
        if len(self.indices) == 0:
            raise ValueError(f"所选数据为空: split={split}, chemistry={chemistry}")
        stats = np.load(self.cache_dir / "normalization.npz", allow_pickle=False)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)
        self._data = None

    def __len__(self) -> int:
        return len(self.indices)

    def _memmap(self) -> np.memmap:
        # 每个 DataLoader worker 独立打开句柄，避免 Windows spawn 复制整个缓存。
        if self._data is None:
            self._data = np.memmap(
                self.cache_dir / "series.f32",
                dtype="<f4",
                mode="r",
                shape=(self.total_rows, len(FEATURE_NAMES)),
            )
        return self._data

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        window_id = int(self.indices[item])
        series_id = int(self.window_series[window_id])
        chemistry = int(self.series_chemistry[series_id])
        valid = int(self.window_valid[window_id])
        start = int(self.series_offset[series_id] + self.window_start[window_id])
        x = np.zeros((self.window_length, len(FEATURE_NAMES)), dtype=np.float32)
        x[:valid] = self._memmap()[start : start + valid]
        x[:valid] = (x[:valid] - self.mean[chemistry]) / self.std[chemistry]
        mask = np.arange(self.window_length) < valid
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(int(self.series_label[series_id]), dtype=torch.long),
            "chemistry": torch.tensor(chemistry, dtype=torch.long),
            "vehicle_id": torch.tensor(int(self.series_vehicle[series_id]), dtype=torch.long),
            "series_id": torch.tensor(series_id, dtype=torch.long),
            "window_id": torch.tensor(window_id, dtype=torch.long),
        }

    def close(self) -> None:
        self._data = None
