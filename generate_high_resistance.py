"""LFP 高内阻训练集的独立 TL-WGAN-GP 增强试验。

只读取 processed_v2 的 train VIN；生成文件写入 synthetic/，不改原始 CSV 或缓存。
这是论文思想的多通道卷积原型，不是其 991 点 LSTM GAN 的逐层复现。
"""

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
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from longseq.data import FEATURE_NAMES, LongSequenceDataset
from longseq.model import LongSequenceMamba


class Generator(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean[:6], dtype=torch.float32)[None, None])
        self.register_buffer("std", torch.as_tensor(std[:6], dtype=torch.float32)[None, None])
        self.fc = nn.Linear(128, 128 * 64)
        blocks = []
        for in_channels, out_channels in [(128, 128), (128, 64), (64, 32)]:
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv1d(in_channels, out_channels, 5, padding=2), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*blocks, nn.Conv1d(32, 6, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(len(z), 128, 64)
        raw = (3.0 * torch.tanh(self.net(h).transpose(1, 2))) * self.std + self.mean
        volt = raw[:, :, :3].sort(dim=-1).values
        temp = raw[:, :, 4:6].sort(dim=-1).values
        # 输出顺序：平均电压、最高电压、最低电压、电流、最高温度、最低温度。
        physical = torch.stack((volt[:, :, 1], volt[:, :, 2], volt[:, :, 0],
                                raw[:, :, 3], temp[:, :, 1], temp[:, :, 0]), dim=-1)
        return (physical - self.mean) / self.std


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [6, 32, 64, 96, 128]
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(a, b, 7, stride=2, padding=3), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(128 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2)).flatten(1)).squeeze(-1)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / "data/processed_v2")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "synthetic")
    parser.add_argument("--source-steps", type=int, default=200)
    parser.add_argument("--target-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-critic", type=int, default=2)
    parser.add_argument("--generate", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_windows(ds: LongSequenceDataset, label: int, per_vehicle: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, window_id in enumerate(ds.indices):
        sid = int(ds.window_series[window_id])
        if int(ds.series_label[sid]) == label and int(ds.window_valid[window_id]) == ds.window_length:
            groups[int(ds.series_vehicle[sid])].append(i)
    selected = []
    for vehicle in sorted(groups):
        choices = rng.choice(groups[vehicle], min(per_vehicle, len(groups[vehicle])), replace=False)
        selected.extend(map(int, choices))
    if not selected:
        raise ValueError(f"找不到标签 {label} 的完整训练窗口")
    x = np.stack([ds[i]["x"].numpy()[:, :6] for i in selected]).astype(np.float32)
    return x, sorted(groups)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(len(real), 1, 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    value = critic(mixed)
    grad = torch.autograd.grad(value.sum(), mixed, create_graph=True)[0]
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


def fit_stage(generator: Generator, critic: Critic, data: torch.Tensor, steps: int,
              batch_size: int, n_critic: int, lr: float, name: str) -> list[dict]:
    generator.train()
    critic.train()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    history = []
    for step in tqdm(range(1, steps + 1), desc=name, unit="step"):
        # 先固定生成器的全局种子映射，后期再整体适配少量故障曲线。
        if name == "fault_transfer" and steps > 10:
            generator.fc.requires_grad_(step > steps // 3)
        for _ in range(n_critic):
            idx = torch.randint(len(data), (batch_size,), device=data.device)
            real = data[idx]
            with torch.no_grad():
                fake = generator(torch.randn(batch_size, 128, device=data.device))
            opt_d.zero_grad(set_to_none=True)
            loss_d = critic(fake).mean() - critic(real).mean() + 10 * gradient_penalty(critic, real, fake)
            loss_d.backward()
            opt_d.step()
        opt_g.zero_grad(set_to_none=True)
        fake = generator(torch.randn(batch_size, 128, device=data.device))
        loss_g = -critic(fake).mean()
        loss_g.backward()
        opt_g.step()
        if step == 1 or step % 25 == 0 or step == steps:
            item = {"stage": name, "step": step, "critic_loss": float(loss_d.detach()),
                    "generator_loss": float(loss_g.detach())}
            history.append(item)
            print(json.dumps(item), flush=True)
    generator.fc.requires_grad_(True)
    return history


def denormalize(x6: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw6 = x6 * std[None, None, :6] + mean[None, None, :6]
    spread = raw6[:, :, 1] - raw6[:, :, 2]
    return np.concatenate((raw6, spread[:, :, None]), axis=-1).astype(np.float32)


def summary_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x.mean(1), x.std(1), x[:, 0], x[:, -1],
                           np.abs(np.diff(x, axis=1)).mean(1)), axis=1)


def assess(real: np.ndarray, synthetic: np.ndarray, mean: np.ndarray,
           std: np.ndarray, device: torch.device, classifier_path: Path) -> dict:
    a = denormalize(real, mean, std)
    b = denormalize(synthetic, mean, std)
    checks = {
        "finite": bool(np.isfinite(b).all()),
        "voltage_order": bool(((b[:, :, 2] <= b[:, :, 0]) & (b[:, :, 0] <= b[:, :, 1])).all()),
        "temperature_order": bool((b[:, :, 5] <= b[:, :, 4]).all()),
        "voltage_spread_identity": bool(np.allclose(b[:, :, 6], b[:, :, 1] - b[:, :, 2], atol=1e-6)),
        "lfp_voltage_bounds_2_to_4V": bool(((b[:, :, :3] >= 2) & (b[:, :, :3] <= 4)).all()),
        "current_bounds_250A": bool((np.abs(b[:, :, 3]) <= 250).all()),
        "temperature_bounds_minus20_to_80C": bool(((b[:, :, 4:6] >= -20) & (b[:, :, 4:6] <= 80)).all()),
    }
    feature_names = FEATURE_NAMES
    quantile_gap = {}
    derivative_ratio = {}
    for i, name in enumerate(feature_names):
        ref = np.quantile(a[:, :, i], [0.05, 0.5, 0.95])
        got = np.quantile(b[:, :, i], [0.05, 0.5, 0.95])
        floor = 0.1 if "temperature" in name else 1e-3
        scale = max(float(np.quantile(a[:, :, i], 0.75) - np.quantile(a[:, :, i], 0.25)),
                    float(a[:, :, i].std()), floor)
        quantile_gap[name] = float(np.mean(np.abs(ref - got)) / scale)
        d_ref = float(np.abs(np.diff(a[:, :, i], axis=1)).mean())
        d_new = float(np.abs(np.diff(b[:, :, i], axis=1)).mean())
        derivative_ratio[name] = d_new / max(d_ref, 1e-8)
    ref_summary, gen_summary = summary_features(a), summary_features(b)
    scale = np.maximum(ref_summary.std(0), 1e-5)
    distances = cdist((gen_summary - ref_summary.mean(0)) / scale,
                      (ref_summary - ref_summary.mean(0)) / scale)
    nearest_summary = distances.min(1)
    metrics = {
        "synthetic_count": len(b), "real_train_windows_compared": len(a),
        "physics_checks": checks, "quantile_gap_in_real_iqr": quantile_gap,
        "temporal_derivative_ratio_to_real": derivative_ratio,
        "nearest_real_summary_distance_median": float(np.median(nearest_summary)),
        "nearest_real_summary_distance_p05": float(np.quantile(nearest_summary, 0.05)),
    }
    core = ("mean_cell_voltage", "max_cell_voltage", "min_cell_voltage",
            "pack_current", "cell_voltage_spread")
    distribution_ok = all(quantile_gap[name] <= 0.5 for name in core)
    dynamics_ok = all(0.3 <= derivative_ratio[name] <= 3.0 for name in core)
    metrics["quality_gate"] = {
        "passes_basic_physics": all(checks.values()),
        "passes_distribution": distribution_ok,
        "passes_temporal_dynamics": dynamics_ok,
        "approved_for_classifier_training": all(checks.values()) and distribution_ok and dynamics_ok,
        "note": "探索性筛选阈值；通过也不代表等同新采集的独立车辆",
    }
    if classifier_path.is_file():
        model = LongSequenceMamba().to(device)
        ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        probs = []
        with torch.no_grad():
            for part in np.array_split(synthetic, max(1, (len(synthetic) + 31) // 32)):
                x = np.concatenate((part, (part[:, :, 1] * std[1] + mean[1] -
                                            part[:, :, 2] * std[2] - mean[2] - mean[6])[:, :, None] /
                                            std[6]), axis=-1).astype(np.float32)
                tensor = torch.as_tensor(x, device=device)
                probs.append(model(tensor, torch.ones(tensor.shape[:2], dtype=torch.bool,
                                                      device=device)).softmax(-1).cpu().numpy())
        p = np.concatenate(probs)
        metrics["baseline_classifier_fraction_predict_high_resistance"] = float((p.argmax(1) == 2).mean())
        metrics["baseline_classifier_mean_high_resistance_probability"] = float(p[:, 2].mean())
    return metrics


def main() -> None:
    args = arguments()
    if args.smoke_test:
        args.source_steps, args.target_steps, args.generate = 2, 2, 8
    if args.source_steps < 1 or args.target_steps < 1 or args.generate < 1:
        raise ValueError("步数和生成数量必须为正数")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("本试验需要 CUDA；避免在 CPU 上长时间运行 WGAN-GP")
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    ds = LongSequenceDataset(args.cache, "train", "LFP")
    source, source_vehicles = load_windows(ds, 0, 8, rng)
    target, target_vehicles = load_windows(ds, 2, 64, rng)
    if set(source_vehicles) & set(target_vehicles):
        raise RuntimeError("正常与故障 VIN 重叠")
    mean, std = ds.mean[0].copy(), ds.std[0].copy()
    ds.close()
    run = args.output_root / f"{datetime.now():%Y%m%d_%H%M%S}_LFP_high_resistance{'_smoke' if args.smoke_test else ''}"
    run.mkdir(parents=True, exist_ok=False)
    config = {"cache": str(args.cache.resolve()), "split": "train_only", "chemistry": "LFP",
              "target_class": "high_resistance", "source_vehicles": len(source_vehicles),
              "target_vehicles": len(target_vehicles), "source_windows": len(source),
              "target_windows": len(target), "source_steps": args.source_steps,
              "target_steps": args.target_steps, "batch_size": args.batch_size,
              "n_critic": args.n_critic, "generated": args.generate, "seed": args.seed,
              "smoke_test": args.smoke_test,
              "method": "two-stage convolutional TL-WGAN-GP; paper-inspired, not exact LSTM reproduction"}
    (run / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    generator = Generator(mean, std).to(device)
    critic = Critic().to(device)
    source_tensor = torch.as_tensor(source, device=device)
    target_tensor = torch.as_tensor(target, device=device)
    history = fit_stage(generator, critic, source_tensor, args.source_steps, args.batch_size,
                        args.n_critic, 1e-4, "normal_pretrain")
    history += fit_stage(generator, critic, target_tensor, args.target_steps, args.batch_size,
                         args.n_critic, 5e-5, "fault_transfer")
    generator.eval()
    with torch.no_grad():
        pieces = [generator(torch.randn(min(32, args.generate - i), 128, device=device)).cpu().numpy()
                  for i in range(0, args.generate, 32)]
    synthetic = np.concatenate(pieces).astype(np.float32)
    raw = denormalize(synthetic, mean, std)
    np.savez_compressed(run / "synthetic_only.npz", x_raw=raw, label=np.full(len(raw), 2, dtype=np.int64),
                        feature_names=np.asarray(FEATURE_NAMES), source_split=np.asarray("train_only"))
    torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                "config": config}, run / "generator.pt")
    (run / "losses.jsonl").write_text("".join(json.dumps(item) + "\n" for item in history), encoding="utf-8")
    classifier_path = Path(__file__).resolve().parent / "experiments/20260919_135510_LFP_long_mamba/best.pt"
    quality = assess(target, synthetic, mean, std, device, classifier_path)
    (run / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    real_raw = denormalize(target[:1], mean, std)[0]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    for ax, i in zip(axes.flat, [0, 1, 2, 3, 4, 6]):
        ax.plot(real_raw[:, i], label="real train", linewidth=1.2)
        ax.plot(raw[0, :, i], label="synthetic", linewidth=1.2, alpha=0.8)
        ax.set_title(FEATURE_NAMES[i])
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(run / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"run": str(run.resolve()), "quality": quality}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
