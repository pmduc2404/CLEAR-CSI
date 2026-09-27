"""Rate-distortion result aggregation for fixed-budget models."""

import csv


def effective_compression_ratio(original_dimension, average_latent_dimension):
    """Return compressed/original rate, e.g. 1024/2048 == 0.5."""
    return average_latent_dimension / original_dimension


def write_rate_distortion_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
