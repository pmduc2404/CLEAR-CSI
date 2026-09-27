"""Multi-TRP CSI feedback over heterogeneous, blockage-prone uplinks.

The UE compresses CSI with a Mamba encoder and splits a fixed budget of K real
channel uses over three uplinks (one per TRP). Uplink l has SNR gamma_l and is
blocked with probability p_l; blocked links deliver nothing. A central unit
(CPU) fuses whatever arrives and reconstructs the CSI.

CLEAR extensions (all optional, see MambaMultiLinkCSI): physical transmit power
normalization, a closed-loop CPU decoder that re-encodes its estimate with the
UE chain and back-projects the per-link innovation, and a risk-priced
outage-factorized router over candidate splits.
"""

import copy
import itertools
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_backbone import BiMambaBlock, MambaBackbone

# Link state known at the UE: (SNR / 20 dB, blockage probability) per link.
UE_STATE_DIM = 6
# Link state known at the CPU: (SNR / 20 dB, link arrived, k_l / K) per link.
CPU_STATE_DIM = 9
# The 8 blockage patterns of the three links (1 = blocked), itertools.product order.
OUTAGE_PATTERNS = torch.tensor(list(itertools.product((0.0, 1.0), repeat=3)))
# Measured NMSE when no link that carries data arrives (the decoder sees zeros).
ALL_LOST_NMSE = 1.001


def pattern_probabilities(blockage_prob: torch.Tensor) -> torch.Tensor:
    """[..., 3] independent blockage probabilities -> [..., 8] pattern probabilities."""
    patterns = OUTAGE_PATTERNS.to(blockage_prob)
    p = blockage_prob.unsqueeze(-2)
    return torch.where(patterns.bool(), p, 1.0 - p).prod(dim=-1)


class FiLM(nn.Module):
    """Feature-wise modulation x * (1 + scale(s)) + shift(s); starts as identity."""

    def __init__(self, state_dim: int, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        scale, shift = self.network(state).chunk(2, dim=-1)
        return features * (1.0 + scale) + shift


class ChannelAllocator(nn.Module):
    """Learned split of the budget over the links.

    ``use_content``: the allocator sees the CSI representation (the CPU then needs
    the resulting (k1, k2, k3) as side information).
    ``use_state``: the allocator sees the link state (normalized SNR and blockage
    probability). A state-only allocator needs no side information, because the
    UE and the CPU can both compute it from the link state.
    ``capacity_prior``: logits = alpha * log(expected capacity) + learned residual.
    It starts exactly at the capacity-proportional split (alpha = 1, residual = 0);
    a larger alpha moves towards the greedy split, and the residual learns the
    correction, e.g. spreading the budget when links may be blocked.
    """

    def __init__(self, representation_dim: int = 2048, num_channels: int = 3,
                 hidden_dim: int = 256, use_content: bool = True,
                 use_state: bool = True, capacity_prior: bool = False) -> None:
        super().__init__()
        if not (use_content or use_state):
            raise ValueError("the allocator needs content, state, or both")
        if capacity_prior and not use_state:
            raise ValueError("the capacity prior needs the link state")
        self.use_content = use_content
        self.use_state = use_state
        self.capacity_prior = capacity_prior
        input_dim = ((representation_dim if use_content else 0)
                     + (UE_STATE_DIM if use_state else 0))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )
        if capacity_prior:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.log_alpha = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def forward(self, representation: torch.Tensor, state: torch.Tensor,
                capacity: Optional[torch.Tensor] = None):
        inputs = []
        if self.use_content:
            inputs.append(representation)
        if self.use_state:
            inputs.append(state)
        logits = self.network(torch.cat(inputs, dim=-1))
        if self.capacity_prior:
            logits = logits + self.alpha * torch.log(capacity.clamp_min(1e-3))
        return F.softmax(logits, dim=-1)


class BudgetMask(nn.Module):
    """Prefix mask that keeps exactly k latent values.

    The forward value is always the hard mask, so training and evaluation see the
    same budget. With ``straight_through`` the gradient flows through a sigmoid
    relaxation of the mask so the allocator can still learn.
    """

    def __init__(self, width: int = 512, temperature: float = 8.0) -> None:
        super().__init__()
        self.width = width
        self.temperature = temperature
        self.register_buffer("positions", torch.arange(width, dtype=torch.float32))

    def soft_mask(self, allocation: torch.Tensor, total_budget: int,
                  dtype: torch.dtype) -> torch.Tensor:
        target_dimensions = (allocation * total_budget).clamp(max=self.width)
        boundary = target_dimensions.unsqueeze(1) - self.positions.unsqueeze(0) - 0.5
        return torch.sigmoid(boundary / self.temperature).to(dtype)

    def hard_mask(self, dimensions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (self.positions.unsqueeze(0) < dimensions.unsqueeze(1)).to(dtype)

    def forward(self, allocation: torch.Tensor, dimensions: torch.Tensor,
                total_budget: int, dtype: torch.dtype,
                straight_through: bool = True) -> torch.Tensor:
        hard = self.hard_mask(dimensions, dtype)
        if not straight_through:
            return hard
        soft = self.soft_mask(allocation, total_budget, dtype)
        return hard + soft - soft.detach()


class FeedbackChannel(nn.Module):
    """One analog uplink: optional transmit power normalization, AWGN, blockage.

    ``tx_norm=True`` (physical power constraint): each sample is scaled to unit
    mean power over its transmitted dims before the channel, and the noise has
    variance 1/SNR per dim. The CPU therefore learns nothing about the latent's
    norm. ``tx_norm=False`` (legacy): the noise power is set from each sample's
    own signal power, which leaks that power to the receiver.
    """

    def __init__(self, ideal: bool = False, channel_coefficient: float = 1.0,
                 tx_norm: bool = False) -> None:
        super().__init__()
        self.ideal = ideal
        self.channel_coefficient = channel_coefficient
        self.tx_norm = tx_norm

    @staticmethod
    def normalize(signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Scale each sample to unit mean power over its transmitted dims."""
        eps = torch.finfo(signal.dtype).eps
        active = mask.detach().pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
        power = signal.pow(2).sum(dim=-1, keepdim=True) / active
        return signal * torch.rsqrt(power.clamp_min(eps))

    def transmit(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Noise-free channel input for an already masked signal."""
        if self.tx_norm:
            signal = self.normalize(signal, mask)
        return signal * self.channel_coefficient

    def forward(self, signal: torch.Tensor, snr_db: torch.Tensor,
                outage: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add AWGN only on transmitted dimensions; blocked samples receive nothing.

        ``outage`` is a [B] bool tensor.
        """
        mask = torch.ones_like(signal) if mask is None else mask.detach()
        received = self.transmit(signal, mask)
        if not self.ideal:
            eps = torch.finfo(signal.dtype).eps
            snr_linear = torch.pow(signal.new_tensor(10.0), snr_db / 10.0)
            if self.tx_norm:
                noise_std = torch.rsqrt(snr_linear.clamp_min(eps)).unsqueeze(-1)
            else:
                active = mask.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
                signal_power = received.pow(2).sum(dim=-1, keepdim=True) / active
                noise_power = signal_power / snr_linear.unsqueeze(-1).clamp_min(eps)
                noise_std = noise_power.clamp_min(eps).sqrt()
            received = received + torch.randn_like(signal) * noise_std * mask
        if outage is not None:
            received = received.masked_fill(outage.unsqueeze(-1), 0.0)
        return received


class ModeSelector(nn.Module):
    """Regression router (legacy): predicts each candidate's NMSE from the raw state.

    Kept as a baseline (--router regression). It is miscalibrated for small
    blockage probabilities; CLEAR uses OutageFactorizedRouter.

    With ``use_content`` it also sees the CSI representation; the CPU then needs
    the chosen index as side information (3 bits for 5 candidates). Without it
    the choice depends on the link state only, which the CPU knows as well.
    """

    def __init__(self, num_candidates: int, representation_dim: int = 2048,
                 hidden_dim: int = 256, use_content: bool = False) -> None:
        super().__init__()
        self.use_content = use_content
        input_dim = UE_STATE_DIM + (representation_dim if use_content else 0)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_candidates),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, representation: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # The selector is trained by regression only; it must not shape the code.
        inputs = (torch.cat((state, representation.detach()), dim=-1)
                  if self.use_content else state)
        return F.softplus(self.network(inputs))  # predicted NMSE >= 0


class OutageFactorizedRouter(nn.Module):
    """Risk-priced router over the candidate splits.

    A critic predicts the NMSE of a split for ONE arrival pattern from the
    physical reception configuration (k_l/K, which links arrive, their SNR);
    it never sees p. The expected NMSE of a candidate is the exact sum over the
    8 blockage patterns weighted by their probabilities under the known
    blockage law, so the router extrapolates to unseen p and to other laws. A
    pattern in which no data-carrying link arrives is priced at ALL_LOST_NMSE.
    Trained with the Gamma deviance y * exp(-g) + g (optimum: g = log E[y]).
    """

    FEATURE_DIM = 12

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Law of the arrival patterns: [.., 3] blockage probabilities -> [.., 8]
        # pattern probabilities. Independent links by default; evaluation can
        # plug in another law (e.g. correlated blockage) without retraining.
        self.pattern_law = None

    @staticmethod
    def configuration_features(k_frac: torch.Tensor, snr_db: torch.Tensor,
                               arrived: torch.Tensor) -> torch.Tensor:
        used = (k_frac > 0).to(k_frac.dtype)
        return torch.cat((k_frac, k_frac * arrived, arrived * used, snr_db / 20.0 * used), dim=-1)

    def log_nmse(self, k_frac: torch.Tensor, snr_db: torch.Tensor,
                 arrived: torch.Tensor) -> torch.Tensor:
        features = self.configuration_features(k_frac, snr_db, arrived)
        return self.network(features).squeeze(-1).clamp(-12.0, 5.0)

    def pattern_nmse(self, k_frac: torch.Tensor, snr_db: torch.Tensor,
                     arrived: torch.Tensor) -> torch.Tensor:
        lost = (k_frac * arrived).sum(dim=-1) <= 0
        predicted = self.log_nmse(k_frac, snr_db, arrived).exp()
        return torch.where(lost, torch.full_like(predicted, ALL_LOST_NMSE), predicted)

    def expected_nmse(self, k_frac: torch.Tensor, snr_db: torch.Tensor,
                      blockage_prob: torch.Tensor) -> torch.Tensor:
        """k_frac [B, C, 3]; snr_db, blockage_prob [B, 3] -> expected NMSE [B, C]."""
        batch, num_candidates = k_frac.shape[:2]
        arrived = (1.0 - OUTAGE_PATTERNS.to(k_frac)).view(1, 8, 1, 3).expand(batch, 8, num_candidates, 3)
        k = k_frac.unsqueeze(1).expand(batch, 8, num_candidates, 3)
        snr = snr_db.view(batch, 1, 1, 3).expand(batch, 8, num_candidates, 3)
        per_pattern = self.pattern_nmse(k, snr, arrived)              # [B, 8, C]
        law = self.pattern_law or pattern_probabilities
        weights = law(blockage_prob).unsqueeze(-1)                    # [B, 8, 1]
        return (weights * per_pattern).sum(dim=1)

    def loss(self, k_frac: torch.Tensor, snr_db: torch.Tensor, arrived: torch.Tensor,
             target_nmse: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gamma deviance on observed (split, arrival pattern, NMSE) rows."""
        log_pred = self.log_nmse(k_frac, snr_db, arrived)
        deviance = target_nmse * torch.exp(-log_pred) + log_pred
        valid = ((k_frac * arrived).sum(dim=-1) > 0).to(deviance.dtype)
        if weight is not None:
            valid = valid * weight
        return (deviance * valid).sum() / valid.sum().clamp_min(1.0)


class MultiChannelAggregator(nn.Module):
    """CPU-side fusion of the received links.

    With ``state_dim`` the CPU uses the link state it knows (SNR, which links
    arrived, the split) to reweight each received dimension before fusion and to
    modulate the fused representation, so noisy or missing links can be
    discounted.
    """

    def __init__(self, latent_dim: int = 1536, output_dim: int = 2048,
                 state_dim: int = 0) -> None:
        super().__init__()
        self.projection = nn.Linear(latent_dim, output_dim)
        self.state_dim = state_dim
        if state_dim:
            self.input_film = FiLM(state_dim, latent_dim)
            self.output_film = FiLM(state_dim, output_dim)

    def forward(self, received: Sequence[torch.Tensor],
                state: Optional[torch.Tensor] = None) -> torch.Tensor:
        fused = torch.cat(tuple(received), dim=-1)
        if self.state_dim:
            fused = self.input_film(fused, state)
        fused = self.projection(fused)
        if self.state_dim:
            fused = self.output_film(fused, state)
        return fused

    def linear_part(self, residual: torch.Tensor,
                    state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Jacobian of forward() w.r.t. its input: FiLM scales and W, no shifts or bias."""
        if not self.state_dim:
            return residual @ self.projection.weight.T
        in_scale, _ = self.input_film.network(state).chunk(2, dim=-1)
        out_scale, _ = self.output_film.network(state).chunk(2, dim=-1)
        return ((residual * (1.0 + in_scale)) @ self.projection.weight.T) * (1.0 + out_scale)


class ResidualDenseStack(nn.Module):
    """Extra one-shot CPU capacity (control for the closed loop).

    Each block is x + W2 GELU(W1 LayerNorm(x)) with W2 zero-initialised, so the
    stack starts as the identity. Three 2048-wide blocks cost about 25M MACs,
    roughly the compute of two closed-loop steps.
    """

    def __init__(self, dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            self.blocks.append(block)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            representation = representation + block(representation)
        return representation


class ClosedLoopRefiner(nn.Module):
    """Parameters of the CPU closed-loop (encoder-in-the-loop) decoder.

    The loop itself lives in MambaMultiLinkCSI._refine: the CPU re-encodes its
    estimate with the UE chain, compares it with every arrived link, weights
    the innovation per link by a step that depends on the link state, and
    back-projects it through the aggregator's own linear map.

    ``learned_step``: step = clamp(rule(SNR) + 0.25 tanh(MLP(SNR/20, arrived,
    k/K)), 0, 1), with the MLP zero-initialised (starts at the rule).
    ``refiner``: adds a zero-initialised learned correction R(u, b) computed by
    a bidirectional Mamba over the 32 tokens of [u, b].
    """

    def __init__(self, learned_step: bool = False, refiner: bool = False,
                 d_model: int = 64) -> None:
        super().__init__()
        self.learned_step = learned_step
        if learned_step:
            self.step_network = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.step_network[-1].weight)
            nn.init.zeros_(self.step_network[-1].bias)
        self.has_refiner = refiner
        if refiner:
            self.refiner_input = nn.Linear(2 * d_model, d_model)
            self.refiner_block = BiMambaBlock(d_model)
            self.refiner_output = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.refiner_output.weight)
            nn.init.zeros_(self.refiner_output.bias)

    @staticmethod
    def rule_step(snr_db: torch.Tensor) -> torch.Tensor:
        """0 at <= 5 dB, 0.05 per dB above, capped at 0.75 (reached at 20 dB)."""
        return (0.05 * (snr_db - 5.0)).clamp(0.0, 0.75)

    def step_size(self, snr_db: torch.Tensor, arrived: torch.Tensor,
                  k_frac: torch.Tensor) -> torch.Tensor:
        step = self.rule_step(snr_db)
        if self.learned_step:
            features = torch.stack((snr_db / 20.0, arrived, k_frac), dim=-1)  # [B, 3, 3]
            step = (step + 0.25 * torch.tanh(self.step_network(features).squeeze(-1))).clamp(0.0, 1.0)
        return step * arrived

    def correction(self, representation: torch.Tensor, back_projection: torch.Tensor) -> torch.Tensor:
        if not self.has_refiner:
            return torch.zeros_like(representation)
        tokens = torch.cat((representation.reshape(-1, 32, 64),
                            back_projection.reshape(-1, 32, 64)), dim=-1)
        hidden = self.refiner_block(self.refiner_input(tokens))
        return self.refiner_output(hidden).reshape(representation.shape)


class _UEChainCopy(nn.Module):
    """Separately trained copy of the UE chain for the untied closed-loop control."""

    def __init__(self, owner: "MambaMultiLinkCSI") -> None:
        super().__init__()
        self.backbone = copy.deepcopy(owner.backbone)
        self.ue_film = copy.deepcopy(owner.ue_film)
        self.mode_film = copy.deepcopy(owner.mode_film)
        self.latent_projection = copy.deepcopy(owner.latent_projection)


class MambaMultiLinkCSI(nn.Module):
    """Mamba-based CSI model with a fixed-budget, three-link feedback path.

    Allocation modes:
      equal_3link            K/3 per link
      fixed_3link            fixed fractions (default 0.5/0.3/0.2)
      greedy_capacity        fill links in order of expected capacity
                             (1 - p) * log2(1 + SNR), up to the per-link cap
      proportional_capacity  k_l proportional to expected capacity
      risk_switch            greedy over the safe links (p < risk_threshold) when
                             there is one, otherwise K/3 per link
      multi_link             learned from CSI content + link state (needs side info)
      multi_link_state       learned from link state only (no side info)
      adaptive_wo_snr        learned from CSI content only (ablation)
      mode_select            routes each sample to one of the candidate splits
                             (equal, top-1 / top-2 by expected capacity, top-1 /
                             top-2 over the safest links) from the link state
                             only (no side info)
      mode_select_content    same, choice also sees the CSI (3 bits side info;
                             regression router only)

    Main options (utils/model_args.py holds the CLEAR configuration):
    ``state_conditioning``: the UE-side FiLM sees (SNR, p); the CPU fusion sees
      (SNR, arrived, k/K).
    ``tx_norm``: physical transmit power normalization (see FeedbackChannel).
    ``io_standardize``: standardize the input with data statistics and decode
      with a linear output (clamped to [0, 1] at evaluation).
    ``bidirectional`` / ``token_ffn``: backbone variants.
    ``cpu_extra_layers``: extra residual dense blocks after CPU fusion.
    ``router``: 'regression' (legacy ModeSelector) or 'ofdr'
      (OutageFactorizedRouter) for the select modes.
    ``refine_steps``: closed-loop steps used at evaluation, and during training
      once ``train_refine`` is switched on (see ``_refine``).
    ``refine_step`` ('rule' | 'learned'), ``refine_refiner``,
    ``refine_forward`` ('tied' = the UE chain itself, 'untied' = a separately
      trained copy, 'none' = no re-encoding, the received signal is fed back
      instead), ``refine_detach_forward``: closed-loop variants and controls.

    Select modes: every candidate is decoded (the batch is expanded to B * C
    rows, blockage shared). The codec is trained on sum_c w_c * L_c with
    w_c = (1 - select_floor) * [c selected] + select_floor / C. With
    ``refine_steps`` > 0 in training, only the selected candidate and one
    random other candidate per sample are refined.
    """

    MODES = {"equal_3link", "fixed_3link", "greedy_capacity",
             "proportional_capacity", "multi_link", "multi_link_state",
             "adaptive_wo_snr", "risk_switch", "mode_select", "mode_select_content"}
    LEARNED_MODES = {"multi_link", "multi_link_state", "adaptive_wo_snr"}
    SELECT_MODES = {"mode_select", "mode_select_content"}
    CANDIDATES = ("equal", "top1", "top2", "top1_safe", "top2_safe")
    SELECTION_OVERRIDES = {"risk"} | set(CANDIDATES)

    def __init__(self, d_model: int = 64, mode: str = "multi_link",
                 snr_db: Iterable[float] = (20.0, 10.0, 3.0),
                 blockage_prob: Iterable[float] = (0.0, 0.0, 0.0),
                 feedback_mode: str = "noisy", fixed_allocation: Iterable[float] =
                 (0.5, 0.3, 0.2), total_budget: int = 512,
                 total_latent_dim: int = 1536, per_link_latent_dim: int = 512,
                 mask_temperature: float = 8.0,
                 state_conditioning: bool = True,
                 capacity_prior: bool = True,
                 risk_threshold: float = 0.02,
                 select_floor: float = 1.0,
                 tx_norm: bool = False,
                 io_standardize: bool = False,
                 bidirectional: bool = False,
                 token_ffn: int = 0,
                 cpu_extra_layers: int = 0,
                 router: str = "regression",
                 refine_steps: int = 0,
                 refine_step: str = "rule",
                 refine_refiner: bool = False,
                 refine_forward: str = "tied",
                 refine_detach_forward: bool = False,
                 backbone: str = "mamba") -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown multi-link mode: {mode}")
        if feedback_mode not in {"ideal", "noisy"}:
            raise ValueError("feedback_mode must be 'ideal' or 'noisy'")
        if router not in {"regression", "ofdr"}:
            raise ValueError("router must be 'regression' or 'ofdr'")
        if router == "ofdr" and mode == "mode_select_content":
            raise ValueError("the OFDR router is state-only; use router='regression'")
        if refine_step not in {"rule", "learned"}:
            raise ValueError("refine_step must be 'rule' or 'learned'")
        if refine_forward not in {"tied", "untied", "none"}:
            raise ValueError("refine_forward must be 'tied', 'untied' or 'none'")
        if refine_forward == "none" and not refine_refiner:
            raise ValueError("refine_forward='none' needs refine_refiner=True")
        self.mode = mode
        self.feedback_mode = feedback_mode
        self.representation_dim = 2048
        if total_latent_dim != 3 * per_link_latent_dim:
            raise ValueError("total_latent_dim must equal 3 * per_link_latent_dim")
        if not 0 < total_budget <= total_latent_dim:
            raise ValueError("total_budget must be between 1 and total_latent_dim")
        self.total_latent_dim = total_latent_dim
        self.per_link_latent_dim = per_link_latent_dim
        self.total_budget = total_budget
        self.risk_threshold = risk_threshold
        self.select_floor = select_floor
        self.tx_norm = tx_norm
        self.io_standardize = io_standardize
        self.refine_steps = refine_steps
        self.refine_forward = refine_forward
        self.refine_detach_forward = refine_detach_forward
        self.train_refine = False          # switched on by the Trainer
        self.selection_override: Optional[str] = None
        self.register_buffer("snr_db", torch.tensor(tuple(snr_db), dtype=torch.float32))
        self.register_buffer("blockage_prob",
                             torch.tensor(tuple(blockage_prob), dtype=torch.float32))
        allocation = torch.tensor(tuple(fixed_allocation), dtype=torch.float32)
        if allocation.shape != (3,) or torch.any(allocation < 0) or allocation.sum() <= 0:
            raise ValueError("fixed_allocation must contain three non-negative values")
        self.register_buffer("fixed_allocation", allocation / allocation.sum())
        if io_standardize:
            self.register_buffer("input_mean", torch.full((2, 32, 32), 0.5))
            self.register_buffer("input_std", torch.tensor(1.0))

        if backbone == "mamba":
            backbone = MambaBackbone(d_model=d_model, bidirectional=bidirectional,
                                     token_ffn=token_ffn,
                                     output_activation="linear" if io_standardize else "sigmoid")
        else:
            # Literature codec (CsiNet / CRNet / CLNet / TransNet) with the same 2048-dim interface.
            if io_standardize:
                raise ValueError("io_standardize is only supported with the Mamba backbone")
            from .baseline_backbones import build_backbone
            backbone = build_backbone(backbone)
        self.backbone = backbone
        self.feature_shape = backbone.feature_shape
        self.state_conditioning = state_conditioning
        self.ue_film = FiLM(UE_STATE_DIM, self.representation_dim) if state_conditioning else None
        use_state = mode != "adaptive_wo_snr"
        self.channel_allocator = (
            ChannelAllocator(
                self.representation_dim,
                use_content=mode != "multi_link_state",
                use_state=use_state,
                capacity_prior=capacity_prior and use_state,
            ) if mode in self.LEARNED_MODES else None
        )
        if mode in self.SELECT_MODES:
            num_candidates = len(self.CANDIDATES)
            if router == "ofdr":
                self.mode_selector = OutageFactorizedRouter()
            else:
                self.mode_selector = ModeSelector(num_candidates, self.representation_dim,
                                                  use_content=mode == "mode_select_content")
            self.mode_film = FiLM(UE_STATE_DIM + num_candidates, self.representation_dim)
        else:
            self.mode_selector = None
            self.mode_film = None
        self.latent_projection = nn.Linear(self.representation_dim, total_latent_dim)
        self.budget_mask = BudgetMask(per_link_latent_dim, mask_temperature)
        self.aggregator = MultiChannelAggregator(
            total_latent_dim, self.representation_dim,
            state_dim=CPU_STATE_DIM if state_conditioning else 0,
        )
        self.cpu_extra = (ResidualDenseStack(self.representation_dim, cpu_extra_layers)
                          if cpu_extra_layers else None)
        self.channels = nn.ModuleList(
            FeedbackChannel(ideal=feedback_mode == "ideal", tx_norm=tx_norm) for _ in range(3)
        )
        self.closed_loop = ClosedLoopRefiner(learned_step=refine_step == "learned",
                                             refiner=refine_refiner, d_model=d_model)
        self.cpu_chain = _UEChainCopy(self) if refine_forward == "untied" else None

    @property
    def encoder(self):
        """Compatibility view over the local Mamba encoder layers."""
        return self.backbone.encoder_layers

    @property
    def decoder(self):
        """Compatibility view over the local Mamba decoder layers."""
        return self.backbone.decoder_layers

    # ----------------------------------------------------------------- codec
    def set_io_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.io_standardize:
            raise RuntimeError("io_standardize is off")
        self.input_mean.copy_(mean.reshape(2, 32, 32))
        self.input_std.copy_(torch.as_tensor(std).reshape(()))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std if self.io_standardize else x

    def encode_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_representation(self._standardize(x))

    def decode_representation(self, representation: torch.Tensor) -> torch.Tensor:
        """CPU side: (extra dense blocks) -> Mamba decoder -> CSI estimate."""
        if self.cpu_extra is not None:
            representation = self.cpu_extra(representation)
        output = self.backbone.decode_representation(representation)
        if self.io_standardize:
            output = self.input_mean + self.input_std * output
            if not self.training:
                output = output.clamp(0.0, 1.0)
        return output

    def _ue_features(self, chain: nn.Module, x: torch.Tensor, ue_state: torch.Tensor):
        """Encoder -> UE FiLM. Returns (representation, features)."""
        representation = chain.backbone.encode_representation(self._standardize(x))
        features = representation
        if self.state_conditioning:
            features = chain.ue_film(features, ue_state)
        return representation, features

    def _mode_features(self, chain: nn.Module, features: torch.Tensor, ue_state: torch.Tensor,
                       candidate_index: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(candidate_index, len(self.CANDIDATES)).to(features.dtype)
        return chain.mode_film(features, torch.cat((ue_state, one_hot), dim=-1))

    def _chain_latent(self, chain: nn.Module, x: torch.Tensor, ue_state: torch.Tensor,
                      candidate_index: Optional[torch.Tensor]) -> torch.Tensor:
        """Full UE chain up to the latent (used by the closed loop at the CPU)."""
        _, features = self._ue_features(chain, x, ue_state)
        if candidate_index is not None:
            features = self._mode_features(chain, features, ue_state, candidate_index)
        return chain.latent_projection(features)

    # ------------------------------------------------------------ allocation
    @staticmethod
    def expected_capacity(snr_db: torch.Tensor, blockage_prob: torch.Tensor) -> torch.Tensor:
        """(1 - p) * log2(1 + SNR): bits per channel use that are expected to arrive."""
        return (1.0 - blockage_prob) * torch.log2(1.0 + torch.pow(10.0, snr_db / 10.0))

    def _ordered_allocation(self, capacity: torch.Tensor, num_links: int) -> torch.Tensor:
        """Split K evenly over the ``num_links`` best links (by ``capacity``),
        respecting the per-link cap; leftovers go to the next best links. Ties
        go to the lower link index (stable sort), identically on CPU and GPU."""
        fills, remaining = [], self.total_budget
        share = -(-self.total_budget // num_links)  # ceil(K / num_links)
        for rank in range(3):
            take = min(self.per_link_latent_dim, remaining,
                       share if rank < num_links else remaining)
            fills.append(take)
            remaining -= take
        fills = capacity.new_tensor(fills).expand(capacity.shape[0], -1)
        order = capacity.argsort(dim=-1, descending=True, stable=True)
        return torch.zeros_like(capacity).scatter(1, order, fills) / float(self.total_budget)

    def _greedy_allocation(self, capacity: torch.Tensor) -> torch.Tensor:
        # Best link gets min(cap, K), the next one the rest up to the cap, ...
        return self._ordered_allocation(capacity, num_links=1)

    def _proportional_allocation(self, capacity: torch.Tensor) -> torch.Tensor:
        total = capacity.sum(dim=-1, keepdim=True)
        equal = torch.full_like(capacity, 1.0 / 3.0)
        return torch.where(total > 0, capacity / total.clamp_min(1e-12), equal)

    def candidate_allocations(self, snr_db: torch.Tensor,
                              blockage_prob: torch.Tensor) -> torch.Tensor:
        """[B, C, 3] candidate splits, in the order of ``CANDIDATES``.

        top-k ranks links by expected capacity (1 - p) log2(1 + SNR); top-k_safe
        ranks them by blockage probability first and SNR second, so it never
        prefers a risky link over a safe one (losing a lone link costs NMSE ~ 1).
        """
        capacity = self.expected_capacity(snr_db, blockage_prob)
        rate = torch.log2(1.0 + torch.pow(10.0, snr_db / 10.0))
        safety = rate - 1e3 * blockage_prob
        return torch.stack((
            torch.full_like(capacity, 1.0 / 3.0),
            self._ordered_allocation(capacity, num_links=1),
            self._ordered_allocation(capacity, num_links=2),
            self._ordered_allocation(safety, num_links=1),
            self._ordered_allocation(safety, num_links=2),
        ), dim=1)

    def _allocation(self, representation: torch.Tensor, snr_db: torch.Tensor,
                    blockage_prob: torch.Tensor) -> torch.Tensor:
        batch = representation.shape[0]
        if self.mode == "equal_3link":
            return representation.new_full((batch, 3), 1.0 / 3.0)
        if self.mode == "fixed_3link":
            return self.fixed_allocation.to(representation).expand(batch, -1)
        if self.mode == "risk_switch":
            # Concentrate on the best safe link if there is one (a lost single
            # link costs NMSE ~ 1, so even a few percent risk favours spreading);
            # with no safe link, spread evenly.
            safe = blockage_prob < self.risk_threshold
            rate = torch.log2(1.0 + torch.pow(10.0, snr_db / 10.0))
            greedy = self._greedy_allocation(rate + 1e3 * safe.to(rate.dtype))
            equal = torch.full_like(rate, 1.0 / 3.0)
            return torch.where(safe.any(dim=-1, keepdim=True), greedy, equal)
        if self.mode in ("greedy_capacity", "proportional_capacity"):
            capacity = self.expected_capacity(snr_db, blockage_prob)
            if self.mode == "greedy_capacity":
                return self._greedy_allocation(capacity)
            return self._proportional_allocation(capacity)
        state = torch.cat((snr_db / 20.0, blockage_prob), dim=-1)
        return self.channel_allocator(representation, state,
                                      self.expected_capacity(snr_db, blockage_prob))

    def _override_selection(self, snr_db: torch.Tensor,
                            blockage_prob: torch.Tensor) -> torch.Tensor:
        """Rule-based candidate choice ('risk' or a fixed candidate name)."""
        batch = snr_db.shape[0]
        index = self.CANDIDATES.index
        if self.selection_override == "risk":
            safe = (blockage_prob < self.risk_threshold).any(dim=-1)
            return torch.where(safe, torch.full_like(safe, index("top1_safe"), dtype=torch.long),
                               torch.full_like(safe, index("equal"), dtype=torch.long))
        return torch.full((batch,), index(self.selection_override), dtype=torch.long,
                          device=snr_db.device)

    def _per_sample(self, value: Optional[torch.Tensor], default: torch.Tensor,
                    reference: torch.Tensor, name: str) -> torch.Tensor:
        value = (default if value is None else value).to(reference)
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(reference.shape[0], -1)
        if value.shape != (reference.shape[0], 3):
            raise ValueError(f"{name} must have shape [B, 3] or [3]")
        return value

    def get_hard_allocation(self, allocation: torch.Tensor) -> torch.Tensor:
        """Round allocation fractions to integer dimensions with fixed total budget."""
        if allocation.ndim != 2 or allocation.shape[1] != 3:
            raise ValueError("allocation must have shape [B, 3]")
        raw_dimensions = allocation.detach() * self.total_budget
        hard_dimensions = raw_dimensions.floor().clamp(min=0, max=self.per_link_latent_dim).to(torch.long)
        remainder = self.total_budget - hard_dimensions.sum(dim=-1)
        fractions = raw_dimensions - raw_dimensions.floor()
        priority = fractions + torch.arange(3, device=allocation.device).float().mul(1e-6)
        rows = torch.arange(allocation.shape[0], device=allocation.device)
        for _ in range(self.total_budget):
            pending = remainder > 0
            if not pending.any():
                break
            available = hard_dimensions < self.per_link_latent_dim
            scores = priority.masked_fill(~available, float("-inf"))
            selected = scores.argmax(dim=-1)
            hard_dimensions[rows[pending], selected[pending]] += 1
            # Largest-remainder rounding: a link gets at most one extra dimension
            # per round, so the remainder is spread over the largest fractions.
            priority[rows[pending], selected[pending]] -= 1.0
            remainder = self.total_budget - hard_dimensions.sum(dim=-1)
        if not torch.equal(hard_dimensions.sum(dim=-1),
                           torch.full_like(remainder, self.total_budget)):
            raise RuntimeError("could not project allocation to the fixed budget")
        return hard_dimensions

    # ------------------------------------------------------------ transport
    def _transmit(self, latent: torch.Tensor, allocation: torch.Tensor,
                  hard_allocation: torch.Tensor, snr_db: torch.Tensor,
                  outage: torch.Tensor, straight_through: bool,
                  channel_snr_db: Optional[torch.Tensor] = None):
        """Mask the three latent chunks, send them, and build the CPU state.

        snr_db is the SNR both ends believe (CPU state); channel_snr_db, if given,
        is the true SNR that sets the noise (state-mismatch evaluation).
        """
        channel_snr_db = snr_db if channel_snr_db is None else channel_snr_db
        latent_parts = torch.chunk(latent, 3, dim=-1)
        # Forward always uses the exact integer budget (same in train and eval);
        # straight_through only adds a gradient to a learned allocator.
        masks = tuple(
            self.budget_mask(allocation[:, index], hard_allocation[:, index],
                             self.total_budget, latent.dtype,
                             straight_through=straight_through)
            for index in range(3)
        )
        transmitted_parts = tuple(part * mask for part, mask in zip(latent_parts, masks))
        received = tuple(
            channel(part, channel_snr_db[:, index], outage[:, index], mask)
            for index, (channel, part, mask) in enumerate(
                zip(self.channels, transmitted_parts, masks))
        )
        cpu_state = torch.cat((
            snr_db / 20.0,
            (~outage).to(latent.dtype),
            hard_allocation.to(latent.dtype) / float(self.total_budget),
        ), dim=-1)
        return masks, transmitted_parts, received, cpu_state

    def _expected_reception(self, latent: torch.Tensor,
                            masks: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Noise-free channel input the CPU should see for a latent (masked, normalized)."""
        return [channel.transmit(part * mask.detach(), mask.detach())
                for channel, part, mask in zip(self.channels, torch.chunk(latent, 3, dim=-1), masks)]

    def _refine(self, u0: torch.Tensor, received: Sequence[torch.Tensor],
                masks: Sequence[torch.Tensor], snr_db: torch.Tensor, outage: torch.Tensor,
                hard_allocation: torch.Tensor, cpu_state: torch.Tensor, ue_state: torch.Tensor,
                candidate_index: Optional[torch.Tensor], steps: int) -> List[torch.Tensor]:
        """Closed-loop CPU decoding; returns the estimates after 0..steps steps.

        u_{t+1} = u_t + b_t + R(u_t, b_t), with b_t = J_agg(r_t) and
        r_t,l = step_l * arrived_l * (y_l - F(x_t)_l), where F is the UE chain
        (encoder -> ... -> latent -> prefix mask -> transmit normalization).
        Blocked links and unsent dims contribute nothing. With
        refine_forward='none' (control) r_t,l = step_l * y_l and the update is
        u_{t+1} = u_t + R(u_t, b_t): the refiner sees the reception again but
        there is no re-encoding innovation.
        """
        arrived = (~outage).to(u0.dtype)
        k_frac = hard_allocation.to(u0.dtype) / float(self.total_budget)
        step = self.closed_loop.step_size(snr_db, arrived, k_frac)
        agg_state = cpu_state if self.state_conditioning else None
        chain = self.cpu_chain if self.refine_forward == "untied" else self
        u = u0
        estimates = [self.decode_representation(u0)]
        for _ in range(steps):
            if self.refine_forward == "none":
                residual = torch.cat([step[:, l:l + 1] * received[l] for l in range(3)], dim=-1)
            else:
                latent = self._chain_latent(chain, estimates[-1], ue_state, candidate_index)
                expected = self._expected_reception(latent, masks)
                if self.refine_detach_forward:
                    expected = [e.detach() for e in expected]
                residual = torch.cat([step[:, l:l + 1] * (received[l] - expected[l])
                                      for l in range(3)], dim=-1)
            back_projection = self.aggregator.linear_part(residual, agg_state)
            if self.refine_forward == "none":
                # Control without re-encoding: the fed-back reception is only an
                # input feature of the learned refiner, not an innovation.
                u = u + self.closed_loop.correction(u, back_projection)
            else:
                u = u + back_projection + self.closed_loop.correction(u, back_projection)
            estimates.append(self.decode_representation(u))
        return estimates

    @staticmethod
    def refine_loss_weights(steps: int) -> List[float]:
        """Deep supervision over the refinement steps: 0.2 (one-shot), 0.3, ..., 1.0 (final)."""
        if steps <= 0:
            return [1.0]
        weights = [0.2] + [0.3] * (steps - 1) + [1.0]
        total = sum(weights)
        return [w / total for w in weights]

    def active_refine_steps(self) -> int:
        if self.training:
            return self.refine_steps if self.train_refine else 0
        return self.refine_steps

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, snr_db: Optional[torch.Tensor] = None,
                blockage_prob: Optional[torch.Tensor] = None,
                channel_outage=None,
                return_debug: bool = True, hard_mask: bool = False,
                refine_all_candidates: bool = False,
                force_candidate: Optional[torch.Tensor] = None,
                channel_snr_db: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        snr_db, blockage_prob: [B, 3] or [3]; default to the model buffers.
        channel_snr_db: true channel SNR if it differs from the SNR the UE and CPU
        believe (``snr_db``); used only by the state-mismatch evaluation.
        channel_outage: None samples blockage from ``blockage_prob``; otherwise a
        [B, 3] bool tensor or three booleans forcing which links are blocked.
        refine_all_candidates (select modes, evaluation): refine every candidate,
        so ``candidate_nmse`` reflects the closed-loop decoder for all of them.
        force_candidate (select modes): [B] candidate indices overriding the router.
        """
        batch = x.shape[0]
        snr_db = self._per_sample(snr_db, self.snr_db, x.flatten(1), "snr_db")
        blockage_prob = self._per_sample(blockage_prob, self.blockage_prob,
                                         x.flatten(1), "blockage_prob")
        ue_state = torch.cat((snr_db / 20.0, blockage_prob), dim=-1)
        channel_snr = (snr_db if channel_snr_db is None else
                       self._per_sample(channel_snr_db, self.snr_db, x.flatten(1), "channel_snr_db"))
        representation, features = self._ue_features(self, x, ue_state)
        if channel_outage is None:
            outage = torch.rand_like(blockage_prob) < blockage_prob
        else:
            outage = torch.as_tensor(channel_outage, dtype=torch.bool, device=x.device)
            if outage.ndim == 1:
                if outage.numel() != 3:
                    raise ValueError("channel_outage must contain three booleans")
                outage = outage.unsqueeze(0).expand(batch, -1)
        steps = self.active_refine_steps()
        step_weights = self.refine_loss_weights(steps)
        power = (x - 0.5).flatten(1).pow(2).mean(dim=-1).clamp_min(1e-12)
        extra: Dict = {}

        if self.mode in self.SELECT_MODES:
            num_candidates = len(self.CANDIDATES)
            candidates = self.candidate_allocations(snr_db, blockage_prob)   # [B, C, 3]
            flat_allocation = candidates.reshape(-1, 3)
            flat_hard = self.get_hard_allocation(flat_allocation)
            k_frac = flat_hard.view(batch, num_candidates, 3).to(x.dtype) / float(self.total_budget)
            if isinstance(self.mode_selector, OutageFactorizedRouter):
                predicted_nmse = self.mode_selector.expected_nmse(k_frac, snr_db, blockage_prob)
            else:
                predicted_nmse = self.mode_selector(features, ue_state)
            selected_mode = predicted_nmse.argmin(dim=-1)
            if self.selection_override is not None:
                selected_mode = self._override_selection(snr_db, blockage_prob)
            if force_candidate is not None:
                selected_mode = torch.as_tensor(force_candidate, dtype=torch.long, device=x.device)
            rows = torch.arange(batch, device=x.device)

            # Decode every candidate: B * C rows sharing each sample's blockage.
            candidate_index = torch.arange(num_candidates, device=x.device).repeat(batch)
            ue_rows = ue_state.repeat_interleave(num_candidates, dim=0)
            snr_rows = snr_db.repeat_interleave(num_candidates, dim=0)
            outage_rows = outage.repeat_interleave(num_candidates, dim=0)
            candidate_features = self._mode_features(
                self, features.repeat_interleave(num_candidates, dim=0), ue_rows, candidate_index)
            latent = self.latent_projection(candidate_features)
            masks, transmitted, received, cpu_state = self._transmit(
                latent, flat_allocation, flat_hard, snr_rows, outage_rows, straight_through=False,
                channel_snr_db=channel_snr.repeat_interleave(num_candidates, dim=0))
            u0 = self.aggregator(received, cpu_state if self.state_conditioning else None)
            one_shot = self.decode_representation(u0)                           # [B*C, ...]
            x_rows = x.repeat_interleave(num_candidates, dim=0)
            row_mse = (one_shot - x_rows).pow(2).flatten(1).mean(dim=-1)
            train_row_loss = row_mse
            final = one_shot
            final_u = u0
            refined_rows = None
            if steps > 0:
                if self.training:
                    other = (selected_mode + torch.randint(
                        1, num_candidates, (batch,), device=x.device)) % num_candidates
                    refined_rows = torch.cat((rows * num_candidates + selected_mode,
                                              rows * num_candidates + other))
                elif refine_all_candidates:
                    refined_rows = torch.arange(batch * num_candidates, device=x.device)
                else:
                    refined_rows = rows * num_candidates + selected_mode
                pick = lambda t: t[refined_rows]
                estimates = self._refine(
                    pick(u0), [pick(r) for r in received], [pick(m) for m in masks],
                    pick(snr_rows), pick(outage_rows), pick(flat_hard), pick(cpu_state),
                    pick(ue_rows), pick(candidate_index), steps)
                x_ref = pick(x_rows)
                refined_loss = sum(w * (e - x_ref).pow(2).flatten(1).mean(dim=-1)
                                   for w, e in zip(step_weights, estimates))
                train_row_loss = row_mse.index_copy(0, refined_rows, refined_loss)
                final = one_shot.index_copy(0, refined_rows, estimates[-1])
                row_mse = row_mse.index_copy(
                    0, refined_rows, (estimates[-1] - x_ref).pow(2).flatten(1).mean(dim=-1))
            all_reconstructions = final.view(batch, num_candidates, *x.shape[1:])
            candidate_loss = row_mse.view(batch, num_candidates)
            candidate_nmse = candidate_loss.detach() / power.unsqueeze(1)
            reconstruction = all_reconstructions[rows, selected_mode]
            allocation = candidates[rows, selected_mode]
            hard_allocation = flat_hard.view(batch, num_candidates, 3)[rows, selected_mode]
            loss_rec = F.mse_loss(reconstruction, x)
            loss_route = torch.zeros((), device=x.device)
            if self.training:
                weights = ((1.0 - self.select_floor)
                           * F.one_hot(selected_mode, num_candidates).to(x.dtype)
                           + self.select_floor / num_candidates)
                codec_loss = (weights * train_row_loss.view(batch, num_candidates)).sum(dim=-1).mean()
                if isinstance(self.mode_selector, OutageFactorizedRouter):
                    # Observed (split, arrival pattern) -> NMSE. With refinement
                    # on, only refined rows reflect the decoder in use.
                    route_weight = None
                    if refined_rows is not None:
                        route_weight = torch.zeros(batch * num_candidates, device=x.device)
                        route_weight[refined_rows] = 1.0
                    loss_route = self.mode_selector.loss(
                        k_frac.reshape(-1, 3), snr_rows, (~outage_rows).to(x.dtype),
                        candidate_nmse.reshape(-1), route_weight)
                else:
                    loss_route = F.mse_loss(predicted_nmse, candidate_nmse)
                loss_total = codec_loss + loss_route
            else:
                loss_total = loss_rec
            view_rows = lambda t: t.view(batch, num_candidates, *t.shape[1:])[rows, selected_mode]
            decoded = {
                "latent": view_rows(latent),
                "transmitted_parts": tuple(view_rows(t) for t in transmitted),
                "received_channel_outputs": tuple(view_rows(t) for t in received),
                "decoded_representation": view_rows(final_u),
            }
            extra = {
                "predicted_nmse": predicted_nmse.detach(),
                "selected_mode": selected_mode,
                "candidate_loss": candidate_loss.detach(),
                "candidate_nmse": candidate_nmse,
                "candidate_hard_allocation": flat_hard.view(batch, num_candidates, 3),
                "loss_route": loss_route.detach(),
            }
        else:
            allocation = self._allocation(features, snr_db, blockage_prob)
            hard_allocation = self.get_hard_allocation(allocation)
            latent = self.latent_projection(features)
            masks, transmitted, received, cpu_state = self._transmit(
                latent, allocation, hard_allocation, snr_db, outage,
                straight_through=not hard_mask, channel_snr_db=channel_snr)
            u0 = self.aggregator(received, cpu_state if self.state_conditioning else None)
            if steps > 0:
                estimates = self._refine(u0, received, masks, snr_db, outage, hard_allocation,
                                         cpu_state, ue_state, None, steps)
            else:
                estimates = [self.decode_representation(u0)]
            reconstruction = estimates[-1]
            loss_rec = F.mse_loss(reconstruction, x)
            loss_total = (sum(w * F.mse_loss(e, x) for w, e in zip(step_weights, estimates))
                          if steps > 0 and self.training else loss_rec)
            decoded = {
                "latent": latent,
                "transmitted_parts": transmitted,
                "received_channel_outputs": received,
                "decoded_representation": u0,
                "reconstruction_steps": estimates,
            }

        active_dimensions = hard_allocation.to(reconstruction.dtype)
        loss_rate = active_dimensions.sum(dim=-1).mean() / float(self.total_budget)
        output = {
            "reconstruction": reconstruction,
            "loss_rec": loss_rec,
            "loss_rate": loss_rate,
            "loss_total": loss_total,
        }
        if return_debug:
            output.update({
                "representation": representation,
                "ue_features": features,
                "allocation": allocation,
                "hard_allocation": hard_allocation,
                "active_dimensions": active_dimensions,
                "total_active_dimensions": active_dimensions.sum(dim=-1),
                "snr_db": snr_db,
                "blockage_prob": blockage_prob,
                "outage": outage,
                **decoded,
                **extra,
            })
        return output

    def rate_loss(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return normalized expected active latent dimensions."""
        return output["total_active_dimensions"].mean() / float(self.total_budget)
