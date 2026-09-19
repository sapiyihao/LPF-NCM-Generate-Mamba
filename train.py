"""长时间序列车辆故障四分类；只读取预构建缓存。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from longseq.data import CLASS_NAMES, LongSequenceDataset
from longseq.model import LongSequenceMamba


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / "data" / "processed_v2")
    parser.add_argument("--chemistry", choices=["LFP", "NCM"], required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "experiments")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="仅检查端到端可运行，指标无统计意义")
    return parser.parse_args()


def make_train_sampler(dataset: LongSequenceDataset) -> WeightedRandomSampler:
    """使四类及类内车辆尽量均衡，避免长CSV支配训练。"""
    selected_series = dataset.window_series[dataset.indices]
    labels = dataset.series_label[selected_series]
    vehicles = dataset.series_vehicle[selected_series]
    class_vehicle_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for vehicle in np.unique(vehicles):
        class_vehicle_counts[int(labels[np.flatnonzero(vehicles == vehicle)[0]])] += 1
    vehicle_window_counts = np.bincount(vehicles, minlength=int(dataset.metadata["total_vehicles"]))
    weights = 1.0 / (class_vehicle_counts[labels] * vehicle_window_counts[vehicles])
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             amp: bool, max_batches: int | None = None) -> tuple[dict, np.ndarray]:
    model.eval()
    probs_by_vehicle = defaultdict(list)
    labels_by_vehicle = {}
    window_true, window_pred = [], []
    for batch_no, batch in enumerate(tqdm(loader, desc="评估", unit="batch", leave=False)):
        if max_batches is not None and batch_no >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x, mask)
        probs = logits.float().softmax(dim=-1).cpu().numpy()
        pred = probs.argmax(axis=1)
        truth = batch["label"].numpy()
        vehicle_ids = batch["vehicle_id"].numpy()
        window_true.extend(truth.tolist())
        window_pred.extend(pred.tolist())
        for vid, y, p in zip(vehicle_ids, truth, probs):
            vid = int(vid)
            if vid in labels_by_vehicle and labels_by_vehicle[vid] != int(y):
                raise ValueError("同一车辆出现多个类别标签")
            labels_by_vehicle[vid] = int(y)
            probs_by_vehicle[vid].append(p)
    vehicle_ids = sorted(labels_by_vehicle)
    true = np.asarray([labels_by_vehicle[vid] for vid in vehicle_ids])
    pred = np.asarray([np.mean(probs_by_vehicle[vid], axis=0).argmax() for vid in vehicle_ids])
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    metrics = {
        "vehicle_count": len(vehicle_ids), "window_count": len(window_true),
        "vehicle_accuracy": float(accuracy_score(true, pred)),
        "vehicle_macro_f1": float(f1_score(true, pred, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)),
        "window_macro_f1": float(f1_score(window_true, window_pred, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)),
    }
    return metrics, matrix


def main() -> None:
    args = arguments()
    if args.batch_size < 1 or args.num_workers < 0 or args.epochs < 1:
        raise ValueError("batch-size和epochs须大于0，num-workers不能为负")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train = LongSequenceDataset(args.cache, "train", args.chemistry)
    val = LongSequenceDataset(args.cache, "val", args.chemistry)
    test = LongSequenceDataset(args.cache, "test", args.chemistry)
    if args.smoke_test:
        args.epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                      "pin_memory": amp}
    if args.num_workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train, sampler=make_train_sampler(train), **loader_options)
    val_loader = DataLoader(val, shuffle=False, **loader_options)
    test_loader = DataLoader(test, shuffle=False, **loader_options)
    model = LongSequenceMamba().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = args.output_root / f"{datetime.now():%Y%m%d_%H%M%S}_{args.chemistry}_long_mamba{'_smoke' if args.smoke_test else ''}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "chemistry": args.chemistry, "cache": str(args.cache.resolve()), "batch_size": args.batch_size,
        "num_workers": args.num_workers, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "seed": args.seed, "model": "PyTorch selective SSM (Mamba-style, not official fused kernel)",
        "smoke_test": args.smoke_test,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    best_f1 = -1.0
    history = []
    max_batches = 2 if args.smoke_test else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_no, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")):
            if max_batches is not None and batch_no >= max_batches:
                break
            x = batch["x"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x, mask)
                loss = nn.functional.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        val_metrics, _ = evaluate(model, val_loader, device, amp, max_batches)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(record)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["vehicle_macro_f1"] > best_f1:
            best_f1 = val_metrics["vehicle_macro_f1"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "validation": val_metrics}, run_dir / "best.pt")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, matrix = evaluate(model, test_loader, device, amp, max_batches)
    summary = {"chemistry": args.chemistry, "best_epoch": best["epoch"],
               "best_validation": best["validation"], "test": test_metrics,
               "test_confusion_matrix": matrix.tolist(), "smoke_test": args.smoke_test}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([x["epoch"] for x in history], [x["train_loss"] for x in history], marker="o", label="train loss")
    axes[0].plot([x["epoch"] for x in history], [x["vehicle_macro_f1"] for x in history], marker="o", label="val vehicle macro-F1")
    axes[0].legend()
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=axes[1], colorbar=False, xticks_rotation=30)
    figure.tight_layout()
    figure.savefig(run_dir / "results.png", dpi=170)
    plt.close(figure)
    print(f"实验完成: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
