"""长安 96S1P 单体内阻异常的隔离式 ECM 试验。

保留真实车辆的电流、SOC、温度和基线电压，用两 RC 等效电路计算
单节故障电芯相对于基线电芯的反事实电压变化。不修改原始数据。
源数据无逐电芯故障标签，基线健康只是假设。
这不是论文中的完整 ISEAFrame 电热耦合模拟器，生成数据默认不可训练。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.signal import lfilter


CELL_COLUMNS = [f"cell_{index}" for index in range(1, 97)]
TEMP_COLUMNS = [f"temp_{index}" for index in range(1, 33)]
USE_COLUMNS = ["terminaltime", "soc", "mode", "totalcurrent", "totalvoltage"] + CELL_COLUMNS + TEMP_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("H:/mamba/changanDataset/vin102.csv"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "synthetic/changan_ecm")
    parser.add_argument("--rows", type=int, default=150_000)
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_ecm(frame: pd.DataFrame, train_rows: int) -> tuple[np.ndarray, dict]:
    time = frame.terminaltime.to_numpy(np.float64)
    current = frame.totalcurrent.to_numpy(np.float64)
    soc = frame.soc.to_numpy(np.float64)
    voltage = np.median(frame[CELL_COLUMNS].to_numpy(np.float64), axis=1)
    # 10 s 采样下的固定时间常数是可辨识的近似，不代表实验室 EIS 标定。
    taus = (40.0, 300.0)
    smoothed = []
    for tau in taus:
        a = 1.0 - np.exp(-10.0 / tau)
        smoothed.append(lfilter([a], [1.0, -(1.0 - a)], current))
    design = np.column_stack(
        [-np.diff(current), -np.diff(smoothed[0]), -np.diff(smoothed[1]),
         np.diff(soc), np.ones(len(current) - 1)]
    )
    target = np.diff(voltage)
    step = np.diff(time)
    valid = ((step >= 9) & (step <= 11) & np.isfinite(design).all(axis=1)
             & np.isfinite(target) & (np.abs(target) < 0.2)
             & (np.abs(np.diff(current)) < 350))
    valid[:500] = False
    train = valid & (np.arange(len(valid)) < train_rows)
    test = valid & ~train
    if train.sum() < 5000 or test.sum() < 1000:
        raise ValueError("可用于标定或独立验证的电流阶跃不足")
    fit = lsq_linear(
        design[train], target[train],
        bounds=([0, 0, 0, -0.1, -0.01], [0.01, 0.01, 0.01, 0.1, 0.01]),
    )
    predicted = design[test] @ fit.x
    rmse = float(np.sqrt(np.mean((predicted - target[test]) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(target[test] ** 2)))
    step_test = test & (np.abs(np.diff(current)) > 20)
    step_rmse = float(np.sqrt(np.mean((design[step_test] @ fit.x - target[step_test]) ** 2)))
    resistance = fit.x[:3]
    metrics = {
        "train_pairs": int(train.sum()), "validation_pairs": int(test.sum()),
        "validation_voltage_delta_rmse_V": rmse,
        "validation_zero_predictor_rmse_V": zero_rmse,
        "validation_current_step_rmse_V": step_rmse,
        "r0_r1_r2_mohm": (1000 * resistance).tolist(),
        "tau1_tau2_s_fixed": list(taus),
        "soc_quantization_note": "SOC 作为差分回归的干扰项，并非 OCV-SOC 曲线标定",
    }
    return resistance, metrics


def select_windows(frame: pd.DataFrame, start: int, length: int, count: int) -> list[int]:
    time = frame.terminaltime.to_numpy(np.float64)
    current = frame.totalcurrent.to_numpy(np.float64)
    cuts = np.flatnonzero((np.diff(time) < 9) | (np.diff(time) > 11)) + 1
    left = np.r_[0, cuts]
    right = np.r_[cuts, len(frame)]
    options = []
    for a, b in zip(left, right):
        if a < start or b - a < length:
            continue
        # 不从相同连续片段抽取重叠窗口，避免把重叠当成独立样本。
        position = int(a + (b - a - length) // 2)
        piece = current[position:position + length]
        options.append((float(np.mean(np.abs(piece))), position))
    options.sort(reverse=True)
    if len(options) < count:
        raise ValueError(f"只有 {len(options)} 个非重叠连续片段，少于请求的 {count} 个")
    # 动态负载优先，但保留一个低负载对照。
    selected = [position for _, position in options[:count - 1]]
    selected.append(options[-1][1])
    return selected


def polarization(current: np.ndarray, resistance: float, tau: float, initial: float = 0.0) -> np.ndarray:
    coefficient = np.exp(-10.0 / tau)
    result = np.empty(len(current), dtype=np.float64)
    state = initial
    for index, value in enumerate(current):
        state = coefficient * state + (1.0 - coefficient) * resistance * value
        result[index] = state
    return result


def synthesize(frame: pd.DataFrame, start: int, length: int, resistance: np.ndarray,
               alpha: float, cell_index: int) -> dict[str, np.ndarray | float | int]:
    piece = frame.iloc[start:start + length]
    current = piece.totalcurrent.to_numpy(np.float64)
    normal_cells = piece[CELL_COLUMNS].to_numpy(np.float64)
    fault_cells = normal_cells.copy()
    onset = length // 4
    taus = (40.0, 300.0)
    delta = np.zeros(length, dtype=np.float64)
    extra_heat_w = np.zeros(length, dtype=np.float64)
    for r, tau in zip(resistance[1:], taus):
        base = polarization(current, float(r), tau)
        # 故障发生时电容电压连续，电阻升高但电容保持不变，因此 tau 同比例增加。
        fault = polarization(current[onset:], float(alpha * r), tau * alpha,
                             initial=float(base[onset - 1]))
        delta[onset:] -= fault - base[onset:]
        extra_heat_w[onset:] += (fault ** 2 / (alpha * r)) - (base[onset:] ** 2 / r)
    delta[onset:] -= current[onset:] * (alpha - 1.0) * resistance[0]
    extra_heat_w[onset:] += current[onset:] ** 2 * (alpha - 1.0) * resistance[0]
    fault_cells[:, cell_index] += delta
    normal_pack = normal_cells.sum(axis=1)
    fault_pack = fault_cells.sum(axis=1)
    return {
        "source_start_row": start, "fault_cell_1based": cell_index + 1,
        "alpha": alpha, "onset_step": onset,
        "time_s": piece.terminaltime.to_numpy(np.float64),
        "current_A": current, "soc_percent": piece.soc.to_numpy(np.float64),
        "mode": piece["mode"].to_numpy(np.int16),
        "normal_cells_V": normal_cells, "fault_cells_V": fault_cells,
        "measured_pack_V": piece.totalvoltage.to_numpy(np.float64),
        "normal_cell_sum_pack_V": normal_pack, "fault_cell_sum_pack_V": fault_pack,
        "measured_probes_C": piece[TEMP_COLUMNS].to_numpy(np.float64),
        "fault_voltage_delta_V": delta, "extra_resistive_heat_W": extra_heat_w,
    }


def checks(samples: list[dict], fit_metrics: dict) -> dict:
    pre_error = []
    bounds = []
    sign_agree = []
    heat = []
    drop = []
    voltage_ranges = []
    pack_consistency = []
    for sample in samples:
        n = int(sample["onset_step"])
        normal = sample["normal_cells_V"]
        fault = sample["fault_cells_V"]
        d = sample["fault_voltage_delta_V"]
        i = sample["current_A"]
        pre_error.append(float(np.max(np.abs(fault[:n] - normal[:n]))))
        bounds.append(bool(np.all((fault >= 2.5) & (fault <= 4.3))))
        voltage_ranges.append([float(np.min(fault)), float(np.max(fault))])
        active = np.abs(i[n:]) > 10
        sign_agree.append(float(np.mean(np.sign(d[n:][active]) == -np.sign(i[n:][active]))) if active.any() else None)
        heat.append(float(np.sum(sample["extra_resistive_heat_W"][n:] * 10)))
        drop.append(float(np.max(np.abs(d[n:]))))
        pack_consistency.append(float(np.max(np.abs(sample["fault_cell_sum_pack_V"] -
                                                       sample["normal_cell_sum_pack_V"] - d))))
    valid_sign = [value for value in sign_agree if value is not None]
    verdict = {
        "baseline_dynamic_fit": fit_metrics["validation_voltage_delta_rmse_V"]
        < 0.5 * fit_metrics["validation_zero_predictor_rmse_V"],
        "prefault_identity": max(pre_error) < 1e-10,
        "cell_voltage_2p5_to_4p3_V": all(bounds),
        "current_voltage_direction": bool(valid_sign) and min(valid_sign) >= 0.9,
        "nonnegative_extra_heat_integral": min(heat) >= -1e-6,
        "pack_sum_consistency": max(pack_consistency) < 1e-8,
        "thermal_response_validated": False,
        "external_real_fault_validated": False,
        "baseline_health_verified": False,
        "ready_for_training": False,
    }
    return {"verdict": verdict, "max_prefault_change_V": max(pre_error),
            "max_fault_voltage_change_V": max(drop),
            "fault_cell_voltage_range_by_candidate_V": voltage_ranges,
            "extra_heat_energy_J_range": [min(heat), max(heat)],
            "current_voltage_direction_fraction": sign_agree,
            "max_pack_sum_error_V": max(pack_consistency),
            "notes": [
                "前六项只是内部一致性/基线动态检查，不等于独立真实故障验证。",
                "长安数据无单体温度-探针几何映射及热容/换热标定，故未生成故障温度。",
                "保留原始探针温度作为外生工况；extra_resistive_heat_W 是模型预测而非测量。",
                "样本均来自同一辆车的互不重叠验证片段，不等于八辆独立虚拟车。",
                "2.5–4.3 V 是首轮保守筛查线，不是长安电芯的正式 BMS 安全阈值。",
                "源 CSV 无逐电芯故障真值，基线健康状态未独立确认。",
            ]}


def plot_examples(samples: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for sample in samples[:3]:
        time_min = (sample["time_s"] - sample["time_s"][0]) / 60
        cell = int(sample["fault_cell_1based"]) - 1
        label = f"cell {cell + 1}, alpha={sample['alpha']:.2f}"
        axes[0].plot(time_min, sample["current_A"], alpha=0.8, label=label)
        axes[1].plot(time_min, sample["normal_cells_V"][:, cell], linestyle="--", alpha=0.7)
        axes[1].plot(time_min, sample["fault_cells_V"][:, cell], label=label)
        axes[2].plot(time_min, 1000 * sample["fault_voltage_delta_V"], label=label)
        axes[3].plot(time_min, sample["extra_resistive_heat_W"], label=label)
    for axis, title, unit in zip(axes,
                                 ["Measured current", "Cell voltage: dashed normal, solid fault",
                                  "Fault voltage delta", "Predicted extra resistive heat"],
                                 ["A", "V", "mV", "W"]):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Minutes from window start")
    fig.tight_layout()
    fig.savefig(output / "ecm_examples.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.train_rows >= args.rows or args.count < 2 or args.window < 32:
        raise ValueError("参数无效：标定/验证必须分开，样本数至少 2，窗口至少 32")
    frame = pd.read_csv(args.source, usecols=USE_COLUMNS, nrows=args.rows)
    if len(frame) < args.rows:
        raise ValueError("源 CSV 行数不足")
    resistance, fit_metrics = fit_ecm(frame, args.train_rows)
    starts = select_windows(frame, args.train_rows, args.window, args.count)
    rng = np.random.default_rng(args.seed)
    # 一种故障类型；随机化单体位置和程度，幅度严格限定在论文的 1.5–3 倍。
    samples = [synthesize(frame, start, args.window, resistance,
                          float(rng.uniform(1.5, 3.0)), int(rng.integers(0, 96)))
               for start in starts]
    result = checks(samples, fit_metrics)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_vin102_internal_resistance"
    output = args.output_root / run_name
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "candidates.npz", **{
        key: np.stack([np.asarray(sample[key]) for sample in samples])
        for key in samples[0]
    })
    report = {
        "method": "measured-normal-baseline plus two-RC counterfactual fault; NOT full paper simulator",
        "source": str(args.source), "source_rows_read": len(frame),
        "fit_rows": [0, args.train_rows], "pilot_window_rows": args.window,
        "pilot_windows_from_rows": [args.train_rows, args.rows],
        "independent_vehicle_count": 1, "candidate_count": len(samples),
        "paper_fault": "abnormal internal resistance R0,R1,R2 multiplied by alpha in [1.5,3]",
        "fit": fit_metrics, "quality": result,
        "candidate_details": [{"source_start_row": int(sample["source_start_row"]),
                               "cell": int(sample["fault_cell_1based"]),
                               "alpha": float(sample["alpha"]),
                               "onset_step": int(sample["onset_step"])} for sample in samples],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_examples(samples, output)
    print(json.dumps({"output": str(output), "fit": fit_metrics,
                      "quality": result["verdict"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
