"""Reconstruction metrics and per-sample CSV reports."""

import csv

import torch


def cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    return torch.nn.functional.cosine_similarity(prediction, target, dim=1).mean()


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor):
    centered_prediction = prediction - 0.5
    centered_target = target - 0.5
    error = (centered_prediction - centered_target).flatten(start_dim=1).pow(2).sum(dim=1)
    power = centered_target.flatten(start_dim=1).pow(2).sum(dim=1).clamp_min(torch.finfo(target.dtype).eps)
    nmse = (error / power).mean()
    return {
        "nmse": nmse,
        "nmse_db": 10.0 * torch.log10(nmse.clamp_min(torch.finfo(nmse.dtype).eps)),
        "cosine_similarity": cosine_similarity(centered_prediction, centered_target),
    }


def write_routing_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
