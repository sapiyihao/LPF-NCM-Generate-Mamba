"""检查缓存完整性、划分隔离和批量输入形状。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from longseq.data import CHEMISTRY_NAMES, CLASS_NAMES, SPLIT_NAMES, LongSequenceDataset, load_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / "data" / "processed_v2")
    args = parser.parse_args()
    meta = load_metadata(args.cache)
    arrays = np.load(args.cache / "index.npz", allow_pickle=False)
    expected_bytes = meta["total_rows"] * len(meta["feature_names"]) * 4
    actual_bytes = (args.cache / "series.f32").stat().st_size
    assert actual_bytes == expected_bytes, (actual_bytes, expected_bytes)
    series_vehicle = arrays["series_vehicle"]
    series_split = arrays["series_split"]
    for vehicle in np.unique(series_vehicle):
        assert len(np.unique(series_split[series_vehicle == vehicle])) == 1
    for split in SPLIT_NAMES:
        for chemistry in CHEMISTRY_NAMES:
            ds = LongSequenceDataset(args.cache, split, chemistry)
            sample = ds[0]
            assert sample["x"].shape == (meta["window_length"], len(meta["feature_names"]))
            assert sample["mask"].sum().item() == int(arrays["window_valid"][sample["window_id"]])
            assert bool(sample["x"].isfinite().all())
            batch = next(iter(DataLoader(ds, batch_size=min(4, len(ds)), num_workers=0)))
            print(split, chemistry, "窗口", len(ds), "batch", tuple(batch["x"].shape),
                  "首样本有效时间点", int(sample["mask"].sum()))
            ds.close()
    print("类别:", CLASS_NAMES)
    print("跳过文件:", meta["skipped_files"])
    print("各组车辆:", meta["valid_vehicles_by_group"])
    print("各组窗口:", meta["window_counts_by_group"])
    valid = arrays["window_valid"]
    series = arrays["window_series"]
    chem = arrays["series_chemistry"][series]
    labels = arrays["series_label"][series]
    print("短于完整窗口的比例:")
    for chemistry_id, chemistry in enumerate(CHEMISTRY_NAMES):
        for label_id, label in enumerate(CLASS_NAMES):
            selected = (chem == chemistry_id) & (labels == label_id)
            ratio = float((valid[selected] < meta["window_length"]).mean())
            print(f"  {chemistry}/{label}: {ratio:.1%} ({int(selected.sum())}个窗口)")
    print("检查通过；后续训练只需读取缓存路径，不访问原始CSV。")


if __name__ == "__main__":
    main()
