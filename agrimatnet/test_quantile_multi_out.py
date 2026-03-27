"""
Script di test multi-output per il modello quantile di AgriMatNet.
Salva metriche per target in formato leggibile e predizioni raw in CSV separato.
"""
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_builder.torch_dataset_multi_out import CacheTimeSeriesDatasetMultiOut
from agrimatnet.train import move_to_device, str2bool, masked_mse
from agrimatnet.train_quantile_ablation_mult_out import collate_variable
from agrimatnet.model_quantile_multi_out import AgriMatNetQuantile, quantile_loss
from agrimatnet.train_quantile import parse_quantiles


def compute_std(sum_val, sum_sq, count):
    if count > 1:
        mean_val = sum_val / count
        var = (sum_sq / count) - (mean_val ** 2)
        return math.sqrt(max(var, 0.0))
    return float("nan")


def format_mean_std(mean, std):
    if math.isnan(mean):
        return "nan +- nan"
    if math.isnan(std):
        return f"{mean:.4f} +- nan"
    return f"{mean:.4f} +- {std:.4f}"


def build_target_history_indices(feature_names, target_names):
    indices = {}
    for target_name in target_names:
        feature_name = f"target_history_{target_name}"
        if feature_name not in feature_names:
            raise ValueError(f"Colonna history mancante per il target {target_name}: {feature_name}")
        indices[target_name] = feature_names.index(feature_name)
    return indices




def inverse_transform_multioutput_preds(preds_array, scaler):
    restored = np.asarray(preds_array, dtype=np.float32).copy()
    if scaler is None or not scaler.has_target_stats():
        return restored
    if restored.ndim != 4:
        raise ValueError(f"Shape predizioni non supportata: {restored.shape}")
    for q_idx in range(restored.shape[-1]):
        restored[..., q_idx] = scaler.inverse_transform_target(restored[..., q_idx])
    return restored
def inverse_transform_single_target_history(history_array, history_mask, scaler, target_idx):
    restored = np.asarray(history_array, dtype=np.float32).copy()
    mask = np.asarray(history_mask, dtype=bool)
    valid = (~mask) & np.isfinite(restored)
    if not valid.any() or scaler is None or not scaler.has_target_stats():
        return restored

    if scaler.mode in {"standardization", "standardization_arcsinh"}:
        mean = float(np.asarray(scaler.target_mean, dtype=np.float32).reshape(-1)[target_idx])
        var = float(np.asarray(scaler.target_var, dtype=np.float32).reshape(-1)[target_idx])
        denom = math.sqrt(var + scaler.eps)
        if scaler.mode == "standardization_arcsinh":
            restored[valid] = np.sinh(restored[valid])
        restored[valid] = restored[valid] * denom + mean
    elif scaler.mode == "min-max":
        min_val = float(np.asarray(scaler.target_min_value, dtype=np.float32).reshape(-1)[target_idx])
        max_val = float(np.asarray(scaler.target_max_value, dtype=np.float32).reshape(-1)[target_idx])
        rng = max(max_val - min_val, scaler.eps)
        restored[valid] = restored[valid] * rng + min_val
    else:
        raise ValueError("Modalita` scaler non supportata.")

    return restored


@torch.no_grad()
def evaluate(model, loader, device, quantiles, target_names, history_target_indices, scaler=None):
    model.eval()
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    metrics_by_target = {
        target_name: {
            "total_items": 0,
            "pinball_sum": 0.0,
            "pinball_sum_sq": 0.0,
            "crps_sum": 0.0,
            "crps_sum_sq": 0.0,
            "coverage_hits": 0,
            "coverage_total": 0,
            "sq_error_sum": 0.0,
            "sq_error_sum_sq": 0.0,
            "abs_error_sum": 0.0,
            "abs_error_sum_sq": 0.0,
            "abs_target_sum": 0.0,
            "naive_abs_error_sum": 0.0,
            "naive_count": 0,
            "per_q_sum": np.zeros(len(quantiles), dtype=np.float64),
            "per_q_sum_sq": np.zeros(len(quantiles), dtype=np.float64),
        }
        for target_name in target_names
    }

    rows = []
    sample_plot_data = {name: None for name in target_names}

    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)  # (B, T, K, Q)
        targets = batch["target"]  # (B, T, K)
        mask = batch["target_mask"]  # (B, T, K)

        _ = quantile_loss(preds, targets, mask, quantiles)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)

        if scaler is not None and scaler.has_target_stats():
            preds_np = inverse_transform_multioutput_preds(preds_np, scaler)
            targets_np = scaler.inverse_transform_target(targets_np, mask_np)

        history_target_np_by_name = {}
        history_mask_np_by_name = {}
        for target_name, history_idx in history_target_indices.items():
            history_target_np = batch["history"][:, :, history_idx].detach().cpu().numpy()
            history_mask_np = batch["history_mask"][:, :, history_idx].detach().cpu().numpy().astype(bool)
            if scaler is not None and scaler.has_target_stats():
                target_pos = target_names.index(target_name)
                history_target_np = inverse_transform_single_target_history(
                    history_target_np,
                    history_mask_np,
                    scaler,
                    target_pos,
                )
            history_target_np_by_name[target_name] = history_target_np
            history_mask_np_by_name[target_name] = history_mask_np

        for target_idx, target_name in enumerate(target_names):
            target_preds_np = preds_np[:, :, target_idx, :]
            target_targets_np = targets_np[:, :, target_idx]
            target_mask_np = mask_np[:, :, target_idx]
            valid = ~target_mask_np

            diff = target_targets_np[:, :, None] - target_preds_np
            keep = valid[:, :, None]

            losses = []
            for q_idx, tau in enumerate(quantiles):
                diff_tau = diff[:, :, q_idx]
                loss_tau = np.abs(diff_tau) * (tau * (diff_tau >= 0) + (1 - tau) * (diff_tau < 0))
                loss_tau = loss_tau * keep[..., 0]
                losses.append(loss_tau)

            loss_stack = np.stack(losses, axis=-1)
            per_item_pinball = loss_stack.mean(axis=-1)
            crps_per_item = 2 * np.trapz(loss_stack, quantiles, axis=-1)

            target_metrics = metrics_by_target[target_name]
            batch_items = int(valid.sum())
            target_metrics["total_items"] += batch_items
            target_metrics["pinball_sum"] += float(per_item_pinball.sum())
            target_metrics["pinball_sum_sq"] += float((per_item_pinball ** 2).sum())
            target_metrics["crps_sum"] += float(crps_per_item.sum())
            target_metrics["crps_sum_sq"] += float((crps_per_item ** 2).sum())

            for q_idx in range(len(quantiles)):
                target_metrics["per_q_sum"][q_idx] += float(losses[q_idx].sum())
                target_metrics["per_q_sum_sq"][q_idx] += float((losses[q_idx] ** 2).sum())

            valid = ~target_mask_np
            lower = target_preds_np[..., 0]
            upper = target_preds_np[..., -1]
            within_band = (target_targets_np >= lower) & (target_targets_np <= upper) & valid
            target_metrics["coverage_hits"] += int(within_band.sum())
            target_metrics["coverage_total"] += int(valid.sum())

            preds_median_np = target_preds_np[..., median_idx]
            diff_med = preds_median_np - target_targets_np
            diff_med_masked = diff_med[valid]
            tgt_masked = target_targets_np[valid]
            if diff_med_masked.size > 0:
                sq_errors = diff_med_masked ** 2
                abs_errors = np.abs(diff_med_masked)
                target_metrics["sq_error_sum"] += float(sq_errors.sum())
                target_metrics["sq_error_sum_sq"] += float((sq_errors ** 2).sum())
                target_metrics["abs_error_sum"] += float(abs_errors.sum())
                target_metrics["abs_error_sum_sq"] += float((abs_errors ** 2).sum())
                target_metrics["abs_target_sum"] += float(np.abs(tgt_masked).sum())

            history_target_np = history_target_np_by_name[target_name]
            history_mask_np = history_mask_np_by_name[target_name]
            for sample_idx in range(target_targets_np.shape[0]):
                keep_idx = valid[sample_idx]
                target_vals = target_targets_np[sample_idx][keep_idx]
                if target_vals.size == 0:
                    continue
                hist_valid = history_target_np[sample_idx][~history_mask_np[sample_idx]]
                if hist_valid.size > 0:
                    naive_val = hist_valid[-1]
                    target_metrics["naive_abs_error_sum"] += float(np.abs(target_vals - naive_val).sum())
                    target_metrics["naive_count"] += target_vals.size

                if sample_plot_data[target_name] is None:
                    sample_plot_data[target_name] = {
                        "timestamps": [str(ts) for ts in batch["target_timestamps"][sample_idx]],
                        "targets": target_targets_np[sample_idx][keep_idx],
                        "preds": target_preds_np[sample_idx][keep_idx],
                    }

        batch_size = preds_np.shape[0]
        for sample_idx in range(batch_size):
            row = {
                "area": batch["area"][sample_idx],
                "source": batch["source"][sample_idx],
                "history_start": batch["history_start"][sample_idx],
                "history_end": batch["history_end"][sample_idx],
                "future_start": batch["future_start"][sample_idx],
                "future_end": batch["future_end"][sample_idx],
                "target_timestamps": "|".join(str(ts) for ts in batch["target_timestamps"][sample_idx]),
            }
            for target_idx, target_name in enumerate(target_names):
                target_mask_np = mask_np[sample_idx, :, target_idx]
                target_targets_np = targets_np[sample_idx, :, target_idx]
                row[f"targets_{target_name}"] = "|".join(
                    "" if target_mask_np[j] else f"{target_targets_np[j]:.6f}"
                    for j in range(len(target_targets_np))
                )
                for q_idx, tau in enumerate(quantiles):
                    key = f"pred_{target_name}_q{tau:.2f}".replace(".", "p")
                    row[key] = "|".join(
                        "" if target_mask_np[j] else f"{preds_np[sample_idx, j, target_idx, q_idx]:.6f}"
                        for j in range(preds_np.shape[1])
                    )
            rows.append(row)

    metrics_by_target_formatted = {}
    per_q_metrics = []
    for target_name in target_names:
        m = metrics_by_target[target_name]
        total_items = m["total_items"]
        if total_items > 0:
            pinball_mean = m["pinball_sum"] / total_items
            pinball_std = compute_std(m["pinball_sum"], m["pinball_sum_sq"], total_items)
            crps_mean = m["crps_sum"] / total_items
            crps_std = compute_std(m["crps_sum"], m["crps_sum_sq"], total_items)
            mse = m["sq_error_sum"] / total_items
            mse_std = compute_std(m["sq_error_sum"], m["sq_error_sum_sq"], total_items)
            rmse = math.sqrt(mse)
            rmse_std = 0.5 * mse_std / rmse if rmse > 0 and not math.isnan(mse_std) else float("nan")
            mae = m["abs_error_sum"] / total_items
            mae_std = compute_std(m["abs_error_sum"], m["abs_error_sum_sq"], total_items)
            wmape = m["abs_error_sum"] / m["abs_target_sum"] if m["abs_target_sum"] > 0 else float("nan")
            wmape_std = mae_std * total_items / m["abs_target_sum"] if m["abs_target_sum"] > 0 and not math.isnan(mae_std) else float("nan")
            coverage = m["coverage_hits"] / m["coverage_total"] if m["coverage_total"] > 0 else float("nan")
            coverage_std = math.sqrt(coverage * (1 - coverage)) if m["coverage_total"] > 1 and not math.isnan(coverage) else float("nan")
            if m["naive_count"] > 0 and m["naive_abs_error_sum"] > 0:
                naive_den = m["naive_abs_error_sum"] / m["naive_count"]
                mase = mae / naive_den if naive_den > 0 else float("nan")
                mase_std = mae_std / naive_den if naive_den > 0 and not math.isnan(mae_std) else float("nan")
            else:
                mase = float("nan")
                mase_std = float("nan")
        else:
            pinball_mean = pinball_std = crps_mean = crps_std = float("nan")
            mse = mse_std = rmse = rmse_std = float("nan")
            mae = mae_std = wmape = wmape_std = float("nan")
            coverage = coverage_std = mase = mase_std = float("nan")

        metrics_by_target_formatted[target_name] = [
            ("pinball_loss", format_mean_std(pinball_mean, pinball_std)),
            ("crps", format_mean_std(crps_mean, crps_std)),
            ("coverage", format_mean_std(coverage, coverage_std)),
            ("mse", format_mean_std(mse, mse_std)),
            ("rmse", format_mean_std(rmse, rmse_std)),
            ("mae", format_mean_std(mae, mae_std)),
            ("wmape", format_mean_std(wmape, wmape_std)),
            ("mase", format_mean_std(mase, mase_std)),
        ]
        for q_idx, tau in enumerate(quantiles):
            mean_q = m["per_q_sum"][q_idx] / total_items if total_items > 0 else float("nan")
            std_q = compute_std(m["per_q_sum"][q_idx], m["per_q_sum_sq"][q_idx], total_items) if total_items > 0 else float("nan")
            per_q_metrics.append((target_name, tau, format_mean_std(mean_q, std_q)))

    return metrics_by_target_formatted, per_q_metrics, rows, sample_plot_data


def plot_quantiles(sample_data, quantiles, output_path, target_name):
    if sample_data is None:
        return

    x_idx = np.arange(len(sample_data["targets"]))
    targets = sample_data["targets"]
    preds = sample_data["preds"]
    timestamps = sample_data["timestamps"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(x_idx, targets, color="black", label="Target", zorder=3)
    lower = preds[:, 0]
    upper = preds[:, -1]
    ax.fill_between(x_idx, lower, upper, color="tab:blue", alpha=0.2, label=f"Banda [{quantiles[0]}, {quantiles[-1]}]")
    for q_idx, tau in enumerate(quantiles):
        ax.plot(x_idx, preds[:, q_idx], label=f"q={tau:.2f}")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(timestamps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Valore previsto")
    ax.set_title(f"Previsioni quantili AgriMatNet - {target_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Test probabilistico multi-output di AgriMatNet (quantile).")
    parser.add_argument("--cache-root", required=True, help="Cartella cache (train/val/test).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weights", required=True, help="File .pth con i pesi del modello.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory in cui salvare metriche, predizioni e grafici.")
    parser.add_argument("--no-scaling", action="store_true", help="Disabilita lo scaler salvato nella cache.")
    parser.add_argument("--scaler-path", type=str, default=None, help="Percorso esplicito per lo scaler da usare (es. quello del training).")
    parser.add_argument("--quantiles", type=parse_quantiles, default=None, help="Quantili separati da virgola. Se omesso vengono letti dal checkpoint.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-engineering", type=str2bool, default=False)
    parser.add_argument("--discretize-target", type=str2bool, default=False)
    parser.add_argument("--multi-targets", type=str, default="all", help="Lista target separati da virgola oppure all.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.multi_targets = [x.strip() for x in args.multi_targets.split(",") if x.strip()]

    checkpoint = torch.load(args.weights, map_location="cpu")
    quantiles = args.quantiles
    if quantiles is None:
        stored_args = checkpoint.get("args", {})
        quantiles = stored_args.get("quantiles")
        if quantiles is None:
            raise ValueError("Quantili non specificati ne' nel checkpoint ne' via CLI.")

    dataset = CacheTimeSeriesDatasetMultiOut(
        cache_dir=args.cache_root,
        apply_scaling=not args.no_scaling,
        scaler_path=args.scaler_path,
        feature_engineering=args.feature_engineering,
        discretize_target=args.discretize_target,
        multi_targets=args.multi_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable,
        num_workers=0,
    )

    model = AgriMatNetQuantile(
        input_dim=len(dataset.feature_names),
        quantiles=quantiles,
        num_targets=len(dataset.target_names),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history_target_indices = build_target_history_indices(dataset.feature_names, dataset.target_names)
    metrics_by_target_rows, per_q_metrics, prediction_rows, sample_plot_data = evaluate(
        model,
        loader,
        device,
        quantiles,
        dataset.target_names,
        history_target_indices,
        scaler=dataset.scaler if not args.no_scaling else None,
    )

    experiment_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.weights).resolve().parent
    experiment_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = experiment_dir / "test_predictions_quantile.csv"
    fieldnames = [
        "area",
        "source",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
        "target_timestamps",
    ]
    for target_name in dataset.target_names:
        fieldnames.append(f"targets_{target_name}")
        for tau in quantiles:
            key = f"pred_{target_name}_q{tau:.2f}".replace(".", "p")
            fieldnames.append(key)

    with predictions_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    metrics_path = experiment_dir / "test_metrics_quantile.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        for block_idx, target_name in enumerate(dataset.target_names):
            if block_idx > 0:
                writer.writerow([])
            writer.writerow([target_name, ""])
            for metric_name, metric_value in metrics_by_target_rows[target_name]:
                writer.writerow([metric_name, metric_value])
            for per_q_target_name, tau, metric_value in per_q_metrics:
                if per_q_target_name == target_name:
                    writer.writerow([f"pinball_q{tau:.2f}", metric_value])

    for target_name, sample_data in sample_plot_data.items():
        plot_path = experiment_dir / f"quantile_forecast_{target_name}.png"
        plot_quantiles(sample_data, quantiles, plot_path, target_name)

    print(f"Salvati i risultati in: {predictions_path} e {metrics_path}")
    for target_name in dataset.target_names:
        print(target_name)
        for metric_name, metric_value in metrics_by_target_rows[target_name]:
            print(f"  {metric_name}: {metric_value}")
        for per_q_target_name, tau, metric_value in per_q_metrics:
            if per_q_target_name == target_name:
                print(f"  pinball_q{tau:.2f}: {metric_value}")


if __name__ == "__main__":
    main()
