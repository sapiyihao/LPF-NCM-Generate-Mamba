"""一次性构建 dataset2 长序列缓存，之后训练只读取缓存。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from longseq.data import CHEMISTRY_NAMES, CLASS_NAMES, FEATURE_NAMES, SPLIT_NAMES
from repair_duplicate_split import duplicate_components, assign_components


REQUIRED = [
    "TIME",
    "SUM_VOLTAGE",
    "MAX_CELL_VOLT",
    "MIN_CELL_VOLT",
    "SUM_CURRENT",
    "MAX_TEMP",
    "MIN_TEMP",
    "SOC",
    "CHARGE_STATUS",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "dataset2")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "processed_v1")
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-files", type=int, help="仅用于冒烟测试；正式缓存不要设置")
    return parser.parse_args()


def discover(source: Path, limit_files: int | None) -> tuple[list[dict], list[dict], str]:
    vehicles: list[dict] = []
    files: list[dict] = []
    digest = hashlib.sha256()
    for chemistry in CHEMISTRY_NAMES:
        for label, category in enumerate(CLASS_NAMES):
            category_dir = source / chemistry / category
            if not category_dir.is_dir():
                raise FileNotFoundError(category_dir)
            for vehicle_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                vehicle_id = len(vehicles)
                vehicles.append({"id": vehicle_id, "chemistry": chemistry, "label": label,
                                 "category": category, "vin": vehicle_dir.name})
                for path in sorted(vehicle_dir.glob("*.csv")):
                    stat = path.stat()
                    rel = path.relative_to(source).as_posix()
                    digest.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
                    files.append({"path": path, "relative_path": rel, "vehicle_id": vehicle_id,
                                  "chemistry": CHEMISTRY_NAMES.index(chemistry), "label": label})
    if limit_files is not None:
        if limit_files < 1:
            raise ValueError("--limit-files 必须为正整数")
        # 按化学体系/类别轮流选文件，使小型测试缓存覆盖全部八个分组。
        grouped = defaultdict(list)
        for file in files:
            grouped[(file["chemistry"], file["label"])].append(file)
        for group, group_files in grouped.items():
            by_vehicle = defaultdict(list)
            for file in group_files:
                by_vehicle[file["vehicle_id"]].append(file)
            interleaved = []
            while by_vehicle:
                for vehicle_id in list(by_vehicle):
                    interleaved.append(by_vehicle[vehicle_id].pop(0))
                    if not by_vehicle[vehicle_id]:
                        del by_vehicle[vehicle_id]
            grouped[group] = interleaved
        ordered = []
        while grouped and len(ordered) < limit_files:
            for group in list(grouped):
                if grouped[group]:
                    ordered.append(grouped[group].pop(0))
                if not grouped[group]:
                    del grouped[group]
                if len(ordered) >= limit_files:
                    break
        files = ordered
    return vehicles, files, digest.hexdigest()


def split_vehicles(vehicles: list[dict], seed: int) -> np.ndarray:
    ids = np.arange(len(vehicles))
    strata = np.asarray([f"{v['chemistry']}/{v['label']}" for v in vehicles])
    train, remainder = train_test_split(ids, test_size=0.30, stratify=strata, random_state=seed)
    val, test = train_test_split(remainder, test_size=0.50, stratify=strata[remainder], random_state=seed + 1)
    split = np.empty(len(vehicles), dtype=np.uint8)
    split[train], split[val], split[test] = 0, 1, 2
    return split


def read_csv(path: Path, chemistry: int) -> tuple[np.ndarray | None, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream), [])
    missing = sorted(set(REQUIRED) - set(columns))
    if missing:
        return None, {"reason": "missing_columns", "columns": missing}
    n_cells = sum(column.startswith("VOLT_") for column in columns)
    if not 16 <= n_cells <= 200:
        return None, {"reason": "invalid_cell_count", "n_cells": n_cells}
    try:
        table = pd.read_csv(path, usecols=REQUIRED, encoding="utf-8-sig", low_memory=False)
        numeric = table.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
    except Exception as error:
        return None, {"reason": "read_error", "detail": str(error)[:160]}
    finite = np.isfinite(values).all(axis=1)
    removed = int((~finite).sum())
    values = values[finite]
    if len(values) < 16:
        return None, {"reason": "too_short", "rows": int(len(values))}
    col = {name: i for i, name in enumerate(numeric.columns)}
    time = values[:, col["TIME"]]
    if np.any(np.diff(time) <= 0):
        return None, {"reason": "nonmonotonic_time", "rows": int(len(values))}
    median_dt = float(np.median(np.diff(time)))
    if not 1 <= median_dt <= 120:
        return None, {"reason": "invalid_sampling_interval", "median_dt": median_dt}
    v_mean = values[:, col["SUM_VOLTAGE"]] / n_cells
    v_max = values[:, col["MAX_CELL_VOLT"]]
    v_min = values[:, col["MIN_CELL_VOLT"]]
    if np.any(v_max < v_min) or np.any((values[:, col["SOC"]] < 0) | (values[:, col["SOC"]] > 100)):
        return None, {"reason": "invalid_voltage_or_soc"}
    lower, upper = (2.3, 4.0) if chemistry == 0 else (2.5, 4.5)
    if np.median(v_mean) < lower or np.median(v_mean) > upper:
        return None, {"reason": "implausible_pack_voltage", "median_mean_cell_voltage": float(np.median(v_mean))}
    x = np.stack([
        v_mean, v_max, v_min, values[:, col["SUM_CURRENT"]],
        values[:, col["MAX_TEMP"]], values[:, col["MIN_TEMP"]], v_max - v_min,
    ], axis=1).astype("<f4")
    if not np.isfinite(x).all():
        return None, {"reason": "float32_overflow"}
    info = {"n_cells": n_cells, "rows": int(len(x)), "removed_nonfinite_rows": removed,
            "median_dt_s": median_dt, "mode": "discharge" if "discharge" in path.name.lower() else "charge"}
    return x, info


def starts_for(length: int, window_length: int, stride: int) -> list[int]:
    if length <= window_length:
        return [0]
    starts = list(range(0, length - window_length + 1, stride))
    last = length - window_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def build(args: argparse.Namespace) -> Path:
    if args.window_length < 16 or args.stride < 1:
        raise ValueError("窗口长度至少16，步长必须大于0")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出已存在，禁止覆盖: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"发现未完成的构建目录，请先检查: {staging}")
    vehicles, files, fingerprint = discover(source, args.limit_files)
    split = split_vehicles(vehicles, args.seed)
    staging.mkdir(parents=True)
    series_rows = []
    series_offset, series_length, series_label = [], [], []
    series_chemistry, series_split, series_vehicle = [], [], []
    window_series, window_start, window_valid = [], [], []
    sums = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(2, dtype=np.int64)
    issues = Counter()
    skipped_examples = []
    n_rows = 0
    with (staging / "series.f32").open("wb") as stream:
        for file_no, record in enumerate(files, start=1):
            x, info = read_csv(record["path"], record["chemistry"])
            if x is None:
                issues[info["reason"]] += 1
                if len(skipped_examples) < 30:
                    skipped_examples.append({"file": record["relative_path"], **info})
                continue
            series_id = len(series_rows)
            vehicle_id = record["vehicle_id"]
            chemistry = record["chemistry"]
            record_split = int(split[vehicle_id])
            x.tofile(stream)
            series_rows.append({"id": series_id, "file": record["relative_path"],
                                "vehicle_id": vehicle_id, "split": SPLIT_NAMES[record_split],
                                "chemistry": CHEMISTRY_NAMES[chemistry],
                                "label": CLASS_NAMES[record["label"]], **info})
            series_offset.append(n_rows)
            series_length.append(len(x))
            series_label.append(record["label"])
            series_chemistry.append(chemistry)
            series_split.append(record_split)
            series_vehicle.append(vehicle_id)
            for start in starts_for(len(x), args.window_length, args.stride):
                window_series.append(series_id)
                window_start.append(start)
                window_valid.append(min(args.window_length, len(x) - start))
            if record_split == 0:
                sums[chemistry] += x.sum(axis=0, dtype=np.float64)
                squares[chemistry] += np.square(x, dtype=np.float64).sum(axis=0)
                counts[chemistry] += len(x)
            n_rows += len(x)
            if file_no % 1000 == 0 or file_no == len(files):
                print(f"CSV {file_no}/{len(files)} | 有效 {len(series_rows)} | 时间点 {n_rows:,} | 窗口 {len(window_series):,}", flush=True)
    if n_rows == 0 or np.any(counts == 0):
        raise RuntimeError("没有足够的有效训练序列，缓存未完成")
    # 某些不同VIN的CSV内容完全相同；仅按VIN分割仍可能产生跨集泄漏。
    # 在最终写入索引前，把这些VIN构成的连通分量分到同一个集合，并重新拟合统计量。
    audit_arrays = {
        "series_offset": np.asarray(series_offset, dtype=np.int64),
        "series_length": np.asarray(series_length, dtype=np.int32),
        "series_label": np.asarray(series_label, dtype=np.uint8),
        "series_chemistry": np.asarray(series_chemistry, dtype=np.uint8),
        "series_split": np.asarray(series_split, dtype=np.uint8),
        "series_vehicle": np.asarray(series_vehicle, dtype=np.int32),
    }
    raw = np.memmap(staging / "series.f32", dtype="<f4", mode="r", shape=(n_rows, len(FEATURE_NAMES)))
    components, duplicate_series = duplicate_components(audit_arrays, raw, len(vehicles))
    if components:
        repaired = assign_components(audit_arrays, components, len(vehicles), args.seed)
        repaired[repaired == 255] = split[repaired == 255]
        split = repaired
        series_split = split[np.asarray(series_vehicle)].tolist()
        for record in series_rows:
            record["split"] = SPLIT_NAMES[int(series_split[record["id"]])]
        sums.fill(0)
        squares.fill(0)
        counts.fill(0)
        for offset, length, chemistry, record_split in zip(series_offset, series_length,
                                                             series_chemistry, series_split):
            if record_split == 0:
                x = raw[offset : offset + length]
                sums[chemistry] += x.sum(axis=0, dtype=np.float64)
                squares[chemistry] += np.square(x, dtype=np.float64).sum(axis=0)
                counts[chemistry] += length
        print(f"重复序列审计: {duplicate_series}段重复，{len(components)}组VIN联动划分", flush=True)
    # Windows上必须显式释放映射句柄，才能原子重命名构建目录。
    raw._mmap.close()
    del raw
    mean = sums / counts[:, None]
    variance = np.maximum(squares / counts[:, None] - mean**2, 1e-12)
    std = np.sqrt(variance)
    np.savez(staging / "normalization.npz", mean=mean.astype("<f4"), std=std.astype("<f4"), count=counts)
    np.savez(staging / "index.npz",
             series_offset=np.asarray(series_offset, dtype=np.int64),
             series_length=np.asarray(series_length, dtype=np.int32),
             series_label=np.asarray(series_label, dtype=np.uint8),
             series_chemistry=np.asarray(series_chemistry, dtype=np.uint8),
             series_split=np.asarray(series_split, dtype=np.uint8),
             series_vehicle=np.asarray(series_vehicle, dtype=np.int32),
             window_series=np.asarray(window_series, dtype=np.int32),
             window_start=np.asarray(window_start, dtype=np.int32),
             window_valid=np.asarray(window_valid, dtype=np.int16))
    with (staging / "series.jsonl").open("w", encoding="utf-8") as stream:
        for record in series_rows:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    vehicles_out = [{**v, "split": SPLIT_NAMES[int(split[v["id"]])]} for v in vehicles]
    (staging / "vehicles.json").write_text(json.dumps(vehicles_out, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded_groups = defaultdict(set)
    for record in series_rows:
        key = (record["chemistry"], record["label"], record["split"])
        loaded_groups[key].add(record["vehicle_id"])
    manifest = {
        "status": "complete", "format_version": 1,
        "source": str(source), "source_manifest_sha256": fingerprint,
        "seed": args.seed, "split_unit": "chemistry/VIN", "split_ratios": [0.70, 0.15, 0.15],
        "window_length": args.window_length, "stride": args.stride,
        "sampling_note": "原CSV约10秒采样；不跨CSV，短CSV右侧补零并提供mask",
        "feature_names": FEATURE_NAMES, "class_names": CLASS_NAMES,
        "chemistry_names": CHEMISTRY_NAMES,
        "normalization": "仅用对应化学体系的训练车辆原始时间点计算均值和标准差",
        "source_files": len(files), "valid_series": len(series_rows),
        "duplicate_series": duplicate_series,
        "multi_vehicle_duplicate_groups": len(components),
        "skipped_files": dict(issues), "skipped_examples": skipped_examples,
        "total_vehicles": len(vehicles), "total_rows": n_rows,
        "total_windows": len(window_series),
        "valid_vehicles_by_group": {
            "/".join(key): len(value) for key, value in sorted(loaded_groups.items())
        },
        "window_counts_by_group": dict(Counter(
            f"{CHEMISTRY_NAMES[series_chemistry[s]]}/{CLASS_NAMES[series_label[s]]}/{SPLIT_NAMES[series_split[s]]}"
            for s in window_series
        )),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, output)
    print(f"缓存完成: {output} | 序列 {len(series_rows):,} | 时间点 {n_rows:,} | 窗口 {len(window_series):,}")
    return output


if __name__ == "__main__":
    build(arguments())
