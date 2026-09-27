"""Exact-over-blockage evaluation (primary metric of the CLEAR experiments).

For a link state (SNR, p) the expected NMSE is computed exactly over the 8
blockage patterns, E = sum_pi P(pi | p) * NMSE(pi), instead of drawing blockage
events. Channel noise is still random but seeded per (chunk, pattern).

Primary metric: dB of the mean expected (linear) NMSE over random link states
drawn from the training distribution. Secondary: the no-safe-link
distribution, per-SNR-band breakdowns, and fixed grid cells.
"""

import math
from typing import Dict, Optional

import torch

from models.mamba_multi_link_csi import OUTAGE_PATTERNS, pattern_probabilities

# SNR bands of the strongest link (scheme-independent, so every scheme has the same samples per band).
SNR_BANDS = ((0.0, 10.0), (10.0, 14.0), (14.0, 17.0), (17.0, 20.0001))

GRID_SNR = {
    'equal_20_20_20': (20.0, 20.0, 20.0),
    'heterogeneous_20_10_3': (20.0, 10.0, 3.0),
    'reversed_3_10_20': (3.0, 10.0, 20.0),
    'two_weak_one_strong': (5.0, 5.0, 20.0),
    '17_4_11': (17.0, 4.0, 11.0),
    '6_14_2': (6.0, 14.0, 2.0),
}
GRID_BLOCKAGE = {
    'p0': (0.0, 0.0, 0.0),
    'p005_all': (0.05, 0.05, 0.05),
    'p02_all': (0.2, 0.2, 0.2),
    'p05_link1': (0.5, 0.0, 0.0),
    'p01_link1': (0.1, 0.0, 0.0),
    'p02_link1': (0.2, 0.0, 0.0),
}
MAIN12_SNR = ('equal_20_20_20', 'heterogeneous_20_10_3', 'reversed_3_10_20')
MAIN12_BLOCKAGE = ('p0', 'p005_all', 'p02_all', 'p05_link1')


def to_db(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-12))


def sample_states(num: int, seed: int, distribution: str = 'train', max_blockage: float = 0.3,
                  safe_link_prob: float = 0.5, risk_threshold: float = 0.02):
    """Random link states. 'train': the training distribution (SNR ~ U(0, 20) dB;
    each link safe with prob. safe_link_prob, else p ~ U(0, max_blockage)).
    'nosafe': every link risky, p ~ U(risk_threshold, max_blockage)."""
    generator = torch.Generator().manual_seed(seed)
    snr_db = torch.rand(num, 3, generator=generator) * 20.0
    if distribution == 'train':
        blockage = torch.rand(num, 3, generator=generator) * max_blockage
        safe = torch.rand(num, 3, generator=generator) < safe_link_prob
        blockage = blockage.masked_fill(safe, 0.0)
    elif distribution == 'nosafe':
        blockage = risk_threshold + torch.rand(num, 3, generator=generator) * (max_blockage - risk_threshold)
    else:
        raise ValueError(f'unknown state distribution: {distribution}')
    return snr_db, blockage


@torch.no_grad()
def pattern_nmse(model, x: torch.Tensor, snr_db: torch.Tensor, blockage_prob: torch.Tensor,
                 batch_size: int = 500, noise_seed: int = 0,
                 refine_steps: Optional[int] = None,
                 channel_snr_db: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Per-row NMSE (linear) under each of the 8 forced blockage patterns.

    x [N, 2, 32, 32]; snr_db, blockage_prob [N, 3] (CPU or device tensors): the
    link state the model is given. channel_snr_db [N, 3], if given, is the true
    SNR of the channel (state-mismatch evaluation).
    Returns 'nmse' [N, 8], 'dims' [N, 3] (split actually used), and for the
    select modes 'selected' [N].
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    saved_steps = model.refine_steps
    if refine_steps is not None:
        model.refine_steps = refine_steps
    patterns = OUTAGE_PATTERNS.bool().to(device)
    nmse, dims, selected = [], [], []
    try:
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device)
            sb = snr_db[start:start + batch_size].to(device)
            pb = blockage_prob[start:start + batch_size].to(device)
            cb = None if channel_snr_db is None else channel_snr_db[start:start + batch_size].to(device)
            power = (xb - 0.5).flatten(1).pow(2).mean(dim=-1).clamp_min(1e-12)
            per_pattern = []
            for k in range(8):
                torch.manual_seed(noise_seed * 1_000_003 + start * 8 + k)
                outage = patterns[k].unsqueeze(0).expand(xb.shape[0], -1)
                output = model(xb, snr_db=sb, blockage_prob=pb, channel_outage=outage, hard_mask=True,
                               channel_snr_db=cb)
                error = (output['reconstruction'] - xb).pow(2).flatten(1).mean(dim=-1)
                per_pattern.append((error / power).cpu())
                if k == 0:
                    dims.append(output['active_dimensions'].cpu())
                    if 'selected_mode' in output:
                        selected.append(output['selected_mode'].cpu())
            nmse.append(torch.stack(per_pattern, dim=-1))
    finally:
        model.refine_steps = saved_steps
        model.train(was_training)
    result = {'nmse': torch.cat(nmse), 'dims': torch.cat(dims)}
    if selected:
        result['selected'] = torch.cat(selected)
    return result


def expected_nmse(per_pattern: torch.Tensor, blockage_prob: torch.Tensor, law=None) -> torch.Tensor:
    """[N, 8] per-pattern NMSE and [N, 3] blockage probabilities -> [N] expected NMSE.

    law maps blockage probabilities to pattern probabilities (default: independent links)."""
    law = law or pattern_probabilities
    return (law(blockage_prob.cpu().double()) * per_pattern.double()).sum(dim=-1)


def band_breakdown(expected: torch.Tensor, snr_db: torch.Tensor) -> Dict[str, dict]:
    """dB of mean expected NMSE per SNR band of the strongest link, max_l SNR_l."""
    strongest = snr_db.cpu().max(dim=-1).values
    out = {}
    for low, high in SNR_BANDS:
        mask = (strongest >= low) & (strongest < high)
        key = f'{low:g}-{min(high, 20.0):g}dB'
        out[key] = {'n': int(mask.sum()),
                    'nmse_db': to_db(expected[mask].mean()) if mask.any() else None}
    return out


def evaluate_states(model, x: torch.Tensor, snr_db: torch.Tensor, blockage_prob: torch.Tensor,
                    batch_size: int = 500, noise_seed: int = 0,
                    refine_steps: Optional[int] = None) -> Dict:
    """Exact expected NMSE per row plus summaries."""
    result = pattern_nmse(model, x, snr_db, blockage_prob, batch_size, noise_seed, refine_steps)
    expected = expected_nmse(result['nmse'], blockage_prob)
    conditional = result['nmse'].double().mean(dim=0)  # mean NMSE given each forced pattern
    summary = {
        'nmse_db': to_db(expected.mean()),
        'n': int(expected.numel()),
        'bands': band_breakdown(expected, snr_db),
        'pattern_nmse_db': {''.join('x' if b else 'o' for b in OUTAGE_PATTERNS[k].bool().tolist()):
                            to_db(conditional[k]) for k in range(8)},
    }
    if 'selected' in result:
        counts = torch.bincount(result['selected'], minlength=len(model.CANDIDATES)).double()
        summary['selected_mode_share'] = dict(zip(model.CANDIDATES, (counts / counts.sum()).tolist()))
    return {'expected': expected, 'summary': summary, **result}


def in_distribution(model, data: torch.Tensor, num_samples: int, states_per_sample: int, seed: int,
                    distribution: str = 'train', max_blockage: float = 0.3,
                    safe_link_prob: float = 0.5, risk_threshold: float = 0.02,
                    batch_size: int = 500, refine_steps: Optional[int] = None) -> Dict:
    """The first num_samples of ``data``, each with ``states_per_sample`` random states."""
    x = data[:num_samples].repeat_interleave(states_per_sample, dim=0)
    snr_db, blockage = sample_states(x.shape[0], seed, distribution, max_blockage,
                                     safe_link_prob, risk_threshold)
    result = evaluate_states(model, x, snr_db, blockage, batch_size, noise_seed=seed,
                             refine_steps=refine_steps)
    result['snr_db'], result['blockage_prob'] = snr_db, blockage
    return result


def grid(model, data: torch.Tensor, num_samples: int, cells=None, batch_size: int = 500,
         seed: int = 0, refine_steps: Optional[int] = None) -> Dict:
    """Fixed (SNR, p) cells on the first num_samples of ``data``."""
    x = data[:num_samples]
    if cells is None:
        cells = [(s, b) for s in GRID_SNR for b in GRID_BLOCKAGE]
    out = {}
    for snr_name, blockage_name in cells:
        snr_db = torch.tensor(GRID_SNR[snr_name]).expand(x.shape[0], -1)
        blockage = torch.tensor(GRID_BLOCKAGE[blockage_name]).expand(x.shape[0], -1)
        result = evaluate_states(model, x, snr_db, blockage, batch_size, noise_seed=seed,
                                 refine_steps=refine_steps)
        out[f'{snr_name}|{blockage_name}'] = {'expected': result['expected'], **result['summary']}
    return out


def grid_means(cells: Dict[str, dict]) -> Dict[str, Optional[float]]:
    """Mean of cell dB values and dB of the mean linear NMSE, for main12 and all cells."""
    def means(keys):
        keys = [k for k in keys if k in cells]
        if not keys:
            return None, None
        mean_of_db = sum(cells[k]['nmse_db'] for k in keys) / len(keys)
        db_of_mean = to_db(sum(float(cells[k]['expected'].mean()) for k in keys) / len(keys))
        return mean_of_db, db_of_mean
    main12 = [f'{s}|{b}' for s in MAIN12_SNR for b in MAIN12_BLOCKAGE]
    m12, m12_lin = means(main12)
    allc, allc_lin = means(list(cells))
    return {'main12_mean_db': m12, 'main12_db_of_mean': m12_lin,
            'all_mean_db': allc, 'all_db_of_mean': allc_lin, 'num_cells': len(cells)}
