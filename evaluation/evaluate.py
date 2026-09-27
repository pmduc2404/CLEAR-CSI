"""Evaluation reports for fixed-budget multi-link Mamba models."""

import csv
import json
import os

import torch

from utils.statics import evaluator

from .adaptive_metrics import reconstruction_metrics
from .rate_distortion import effective_compression_ratio


def evaluate_model(model, data_loader, device, output_dir, fixed_cr=None,
                   snr_db=None, blockage_prob=None):
    """Evaluate on ``data_loader`` at fixed uplink SNRs and blockage probabilities.

    Blockage events are drawn per sample from ``blockage_prob``, so NMSE is the
    expected NMSE over blockage. ``channel_robustness_nmse_db`` additionally
    forces every outage pattern.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    rows = []
    nmse_sum = 0.0
    cosine_sum = 0.0
    rate_values = []
    allocation_values = []
    active_dimension_values = []
    metric_sample_count = 0
    rho_sum = 0.0
    rho_sample_count = 0
    outage_values = []
    selected_mode_values = []
    sample_id = 0
    is_multilink = hasattr(model, "total_budget")
    if is_multilink and snr_db is None:
        raise ValueError("snr_db is required for deterministic multi-link evaluation")
    condition = tuple(float(value) for value in snr_db) if snr_db is not None else None
    blockage = tuple(float(value) for value in (blockage_prob or (0.0, 0.0, 0.0)))

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            if is_multilink:
                condition_tensor = torch.tensor(condition, device=device).expand(inputs.shape[0], -1)
                blockage_tensor = torch.tensor(blockage, device=device).expand(inputs.shape[0], -1)
                output = model(inputs, snr_db=condition_tensor,
                               blockage_prob=blockage_tensor, hard_mask=True)
                outage_values.append(output["outage"].detach().cpu())
                if "selected_mode" in output:
                    selected_mode_values.append(output["selected_mode"].detach().cpu())
            else:
                output = model(inputs)
            if isinstance(output, dict):
                prediction = output["reconstruction"]
                rates = output.get(
                    "total_active_dimensions", torch.full((inputs.shape[0],), 2048.0),
                ).detach().cpu().tolist()
                if is_multilink:
                    allocation_values.append(output["allocation"].detach().cpu())
                    active_dimension_values.append(output["active_dimensions"].detach().cpu())
            else:
                prediction = output
                latent_dimension = getattr(getattr(model, "fc_encoder", None), "out_features", 2048)
                rates = [latent_dimension] * inputs.shape[0]

            metrics = reconstruction_metrics(prediction, inputs)
            if len(batch) > 1:
                rho, _, _ = evaluator(prediction, inputs, batch[1].to(device))
                rho_sum += rho.item() * inputs.shape[0]
                rho_sample_count += inputs.shape[0]
            centered_prediction = prediction - 0.5
            centered_inputs = inputs - 0.5
            per_sample_nmse = (
                (centered_prediction - centered_inputs).flatten(start_dim=1).pow(2).sum(dim=1)
                / centered_inputs.flatten(start_dim=1).pow(2).sum(dim=1).clamp_min(torch.finfo(inputs.dtype).eps)
            ).cpu().tolist()
            batch_size = inputs.shape[0]
            nmse_sum += metrics["nmse"].item() * batch_size
            cosine_sum += metrics["cosine_similarity"].item() * batch_size
            metric_sample_count += batch_size
            rate_values.extend(rates)
            if is_multilink:
                outage_cpu = output["outage"].cpu().tolist()
                dims_cpu = output["active_dimensions"].cpu().tolist()
                allocation_cpu = output["allocation"].cpu().tolist()
            for index in range(inputs.shape[0]):
                row = {
                    "sample_id": sample_id,
                    "cr": (rates[index] / 2048.0 if is_multilink
                           else fixed_cr or 2048 // rates[index]),
                    "rate_proxy": rates[index],
                    "nmse": per_sample_nmse[index],
                }
                if condition is not None:
                    row.update({"snr_1": condition[0], "snr_2": condition[1], "snr_3": condition[2]})
                    if is_multilink:
                        row.update({
                            **{f"blocked_{link + 1}": outage_cpu[index][link] for link in range(3)},
                            **{f"active_dims_{link + 1}": dims_cpu[index][link] for link in range(3)},
                            **{f"allocation_{link + 1}": allocation_cpu[index][link] for link in range(3)},
                        })
                rows.append(row)
                sample_id += 1

    total = max(len(rate_values), 1)
    average_rate = sum(rate_values) / total
    summary = {
        "nmse": nmse_sum / max(metric_sample_count, 1),
        "nmse_db": 10.0 * torch.log10(torch.tensor(
            max(nmse_sum / max(metric_sample_count, 1), 1e-12))).item(),
        "rho": rho_sum / rho_sample_count if rho_sample_count else None,
        "cosine_similarity": cosine_sum / max(metric_sample_count, 1),
        "average_latent_dimension": average_rate,
        "effective_compression_ratio": effective_compression_ratio(2048, average_rate),
        "rate_proxy": "latent dimension; no quantization or entropy coding",
        "nmse_definition": "linear NMSE; nmse_db = 10*log10(nmse)",
    }
    if condition is not None:
        summary["snr_db"] = list(condition)
    if is_multilink:
        summary["blockage_prob"] = list(blockage)
        summary["observed_outage_rate"] = torch.cat(outage_values).float().mean(dim=0).tolist()
    if selected_mode_values:
        counts = torch.bincount(torch.cat(selected_mode_values),
                                minlength=len(model.CANDIDATES)).float()
        summary["selected_mode_share"] = dict(zip(model.CANDIDATES,
                                                  (counts / counts.sum()).tolist()))
    if hasattr(model, "mode") and hasattr(model, "snr_db"):
        allocations = torch.cat(allocation_values).mean(dim=0)
        active_dimensions = torch.cat(active_dimension_values)
        summary["allocation"] = allocations.tolist()
        summary["total_budget"] = model.total_budget
        summary["average_active_dimensions"] = active_dimensions.sum(dim=-1).mean().item()
        summary["hard_active_dimensions"] = torch.cat([
            model.get_hard_allocation(allocation_batch)
            for allocation_batch in allocation_values
        ]).float().mean(dim=0).tolist()
        summary["channel_active_dimensions"] = active_dimensions.mean(dim=0).tolist()
        summary["effective_compression_ratio"] = effective_compression_ratio(
            2048, summary["average_active_dimensions"]
        )
        outage_results = {}
        outage_conditions = {
            "all_channels": (False, False, False),
            "channel_1_failure": (True, False, False),
            "channel_2_failure": (False, True, False),
            "channel_3_failure": (False, False, True),
            "channels_1_2_failure": (True, True, False),
            "channels_1_3_failure": (True, False, True),
            "channels_2_3_failure": (False, True, True),
        }
        for condition_name, condition in outage_conditions.items():
            outage_nmse_sum = 0.0
            outage_sample_count = 0
            with torch.no_grad():
                for batch in data_loader:
                    inputs = batch[0].to(device)
                    condition_tensor = torch.tensor(snr_db, device=device).expand(inputs.shape[0], -1)
                    blockage_tensor = torch.tensor(blockage, device=device).expand(inputs.shape[0], -1)
                    output = model(inputs, snr_db=condition_tensor,
                                   blockage_prob=blockage_tensor,
                                   channel_outage=condition, hard_mask=True)
                    outage_nmse_sum += reconstruction_metrics(
                        output["reconstruction"], inputs
                    )["nmse"].item() * inputs.shape[0]
                    outage_sample_count += inputs.shape[0]
            outage_results[condition_name] = 10.0 * torch.log10(
                torch.tensor(max(
                    outage_nmse_sum / max(outage_sample_count, 1), 1e-12
                ))
            ).item()
        summary["channel_robustness_nmse_db"] = outage_results
    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "per_sample_routing.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "rate_distortion.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "average_latent_dimension", "nmse_db"])
        writer.writeheader()
        writer.writerow({"method": model.__class__.__name__,
                         "average_latent_dimension": average_rate,
                         "nmse_db": summary["nmse_db"]})
    return summary
