"""只读取既有缓存，把完全重复的跨VIN序列分到同一个集合。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from longseq.data import CHEMISTRY_NAMES, CLASS_NAMES, FEATURE_NAMES, SPLIT_NAMES, load_metadata


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "data" / "processed_v1")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "processed_v2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def duplicate_components(arrays: dict, raw: np.memmap, vehicle_count: int) -> tuple[list[list[int]], int]:
    parent = np.arange(vehicle_count)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a: int, b: int) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a

    seen = {}
    duplicate_series = 0
    for series_id, (offset, length, vehicle) in enumerate(zip(
        arrays["series_offset"], arrays["series_length"], arrays["series_vehicle"]
    )):
        values = raw[int(offset) : int(offset + length)]
        digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
        earlier = seen.get(digest)
        if earlier is None:
            seen[digest] = (series_id, int(vehicle))
        else:
            duplicate_series += 1
            earlier_series, earlier_vehicle = earlier
            if int(arrays["series_chemistry"][earlier_series]) != int(arrays["series_chemistry"][series_id]):
                raise ValueError("跨化学体系出现相同序列，需人工审查")
            if int(arrays["series_label"][earlier_series]) != int(arrays["series_label"][series_id]):
                raise ValueError("完全相同序列对应不同故障标签，需人工审查")
            union(earlier_vehicle, int(vehicle))
    groups = defaultdict(list)
    for vehicle in range(vehicle_count):
        groups[root(vehicle)].append(vehicle)
    components = [members for members in groups.values() if len(members) > 1]
    return components, duplicate_series


def assign_components(arrays: dict, components: list[list[int]], vehicle_count: int,
                      seed: int) -> np.ndarray:
    split = np.full(vehicle_count, 255, dtype=np.uint8)
    for vehicle, series_split in zip(arrays["series_vehicle"], arrays["series_split"]):
        vehicle = int(vehicle)
        if split[vehicle] not in (255, int(series_split)):
            raise ValueError("原缓存同一车辆跨集合")
        split[vehicle] = int(series_split)
    # 小规模冒烟缓存可能故意只包含部分车辆，未出现的车辆保持255。
    chemistry = np.full(vehicle_count, 255, dtype=np.uint8)
    labels = np.full(vehicle_count, 255, dtype=np.uint8)
    for vehicle, chem, label in zip(arrays["series_vehicle"], arrays["series_chemistry"], arrays["series_label"]):
        chemistry[int(vehicle)] = int(chem)
        labels[int(vehicle)] = int(label)

    # 仅重新分配受到重复VIN影响的化学体系/类别，其他组保持原划分。
    affected = sorted({(int(chemistry[group[0]]), int(labels[group[0]])) for group in components})
    rng = np.random.default_rng(seed)
    for chem, label in affected:
        ids = np.flatnonzero((chemistry == chem) & (labels == label)).tolist()
        grouped = []
        covered = set()
        for group in components:
            if chemistry[group[0]] == chem and labels[group[0]] == label:
                grouped.append(group)
                covered.update(group)
        grouped.extend([[vid] for vid in ids if vid not in covered])
        rng.shuffle(grouped)
        grouped.sort(key=len, reverse=True)
        total = len(ids)
        target = np.array([round(total * 0.70), round(total * 0.15), 0], dtype=np.int64)
        target[2] = total - target[:2].sum()
        filled = np.zeros(3, dtype=np.int64)
        for group in grouped:
            deficits = (target - filled) / np.maximum(target, 1)
            destination = int(np.argmax(deficits))
            split[group] = destination
            filled[destination] += len(group)
    for group in components:
        if len(set(split[group].tolist())) != 1:
            raise AssertionError("重复VIN组没有分在同一集合")
    return split


def main() -> None:
    args = arguments()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"临时构建目录已存在: {staging}")
    meta = load_metadata(source)
    original_meta = dict(meta)
    with np.load(source / "index.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    raw = np.memmap(source / "series.f32", dtype="<f4", mode="r",
                    shape=(int(meta["total_rows"]), len(FEATURE_NAMES)))
    components, duplicate_count = duplicate_components(arrays, raw, int(meta["total_vehicles"]))
    vehicle_split = assign_components(arrays, components, int(meta["total_vehicles"]), args.seed)
    old_cross = sum(len(set(arrays["series_split"][np.isin(arrays["series_vehicle"], group)].tolist())) > 1
                    for group in components)
    arrays["series_split"] = vehicle_split[arrays["series_vehicle"]]

    sums = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(2, dtype=np.int64)
    for offset, length, chem, split in zip(arrays["series_offset"], arrays["series_length"],
                                            arrays["series_chemistry"], arrays["series_split"]):
        if split != 0:
            continue
        x = raw[int(offset) : int(offset + length)]
        chem = int(chem)
        sums[chem] += x.sum(axis=0, dtype=np.float64)
        squares[chem] += np.square(x, dtype=np.float64).sum(axis=0)
        counts[chem] += len(x)
    mean = sums / counts[:, None]
    std = np.sqrt(np.maximum(squares / counts[:, None] - mean**2, 1e-12))
    staging.mkdir(parents=True)
    shutil.copyfile(source / "series.f32", staging / "series.f32")
    np.savez(staging / "index.npz", **arrays)
    np.savez(staging / "normalization.npz", mean=mean.astype("<f4"), std=std.astype("<f4"), count=counts)
    with (source / "series.jsonl").open("r", encoding="utf-8") as reader, \
         (staging / "series.jsonl").open("w", encoding="utf-8") as writer:
        for series_id, line in enumerate(reader):
            record = json.loads(line)
            record["split"] = SPLIT_NAMES[int(arrays["series_split"][series_id])]
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    vehicles = json.loads((source / "vehicles.json").read_text(encoding="utf-8"))
    for vehicle in vehicles:
        vehicle["split"] = SPLIT_NAMES[int(vehicle_split[vehicle["id"]])]
    (staging / "vehicles.json").write_text(json.dumps(vehicles, ensure_ascii=False, indent=2), encoding="utf-8")
    window_series = arrays["window_series"]
    group_vehicle = defaultdict(set)
    window_counts = Counter()
    for series_id, vehicle, chem, label, split in zip(range(len(arrays["series_offset"])),
        arrays["series_vehicle"], arrays["series_chemistry"], arrays["series_label"], arrays["series_split"]):
        key = f"{CHEMISTRY_NAMES[int(chem)]}/{CLASS_NAMES[int(label)]}/{SPLIT_NAMES[int(split)]}"
        group_vehicle[key].add(int(vehicle))
    for series_id in window_series:
        chem, label, split = (int(arrays[key][series_id]) for key in
                              ("series_chemistry", "series_label", "series_split"))
        key = f"{CHEMISTRY_NAMES[chem]}/{CLASS_NAMES[label]}/{SPLIT_NAMES[split]}"
        window_counts[key] += 1
    meta["valid_vehicles_by_group"] = {key: len(value) for key, value in sorted(group_vehicle.items())}
    meta["window_counts_by_group"] = dict(window_counts)
    meta["split_repair"] = {
        "from_cache": str(source), "duplicate_series": duplicate_count,
        "multi_vehicle_duplicate_groups": len(components),
        "cross_split_groups_before": old_cross,
        "components": components,
        "method": "字节级相同序列的VIN连通分量作为同一划分单位；训练统计量重新拟合",
    }
    (staging / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, output)
    # v1 保留以便追溯，但禁止训练入口把它当成有效缓存。
    old_meta = original_meta
    old_meta["status"] = "invalid_cross_split_duplicates"
    old_meta["replaced_by"] = str(output)
    (source / "manifest.json").write_text(json.dumps(old_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"修复完成: {output}; 重复序列 {duplicate_count}; 跨集重复VIN组 {old_cross}; 原始CSV未重新读取")


if __name__ == "__main__":
    main()
