"""Mamba2 backbone for CSI feedback."""

from typing import List, Tuple

import torch
import torch.nn as nn
from mamba_ssm import Mamba2



class MambaBlock(nn.Module):
    """Pre-norm residual Mamba2 block: x + Mamba2(LayerNorm(x))."""

    def __init__(self, d_model: int, state_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = Mamba2(
            d_model=d_model,
            d_state=state_dim,
            headdim=d_model,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.block(self.norm(sequence))


class BiMambaBlock(nn.Module):
    """Pre-norm residual bidirectional Mamba2 block.

    The antenna axis has no causal order, so the block scans it in both
    directions with separate Mamba2 weights and adds the two outputs.
    """

    def __init__(self, d_model: int, state_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_block = Mamba2(d_model=d_model, d_state=state_dim, headdim=d_model)
        self.backward_block = Mamba2(d_model=d_model, d_state=state_dim, headdim=d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(sequence)
        backward = self.backward_block(hidden.flip(1)).flip(1)
        return sequence + self.forward_block(hidden) + backward


class TokenFFN(nn.Module):
    """Pre-norm residual per-token MLP: x + W2 GELU(W1 LayerNorm(x))."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(sequence))))


class MambaBackbone(nn.Module):
    """Encode/decode CSI as a sequence of 32 tokens with width 64.

    Token i is antenna row i: its 32 real values followed by its 32 imaginary
    values. ``bidirectional`` scans the antenna axis both ways; ``token_ffn``
    (hidden size, 0 = off) adds a per-token MLP after every Mamba block;
    ``output_activation`` is 'sigmoid' (data in [0, 1]) or 'linear'.
    """

    def __init__(self, d_model: int = 64, num_layers: int = 2,
                 state_dim: int = 64, bidirectional: bool = False,
                 token_ffn: int = 0, output_activation: str = "sigmoid") -> None:
        super().__init__()
        if d_model != 64:
            raise ValueError("CSI backbone currently requires d_model=64")
        if output_activation not in {"sigmoid", "linear"}:
            raise ValueError("output_activation must be 'sigmoid' or 'linear'")
        self.feature_shape: Tuple[int, int] = (32, d_model)
        self.output_activation = output_activation
        self.encoder_input = nn.Linear(64, d_model)
        self.encoder_layers = nn.ModuleList(
            self._stack(d_model, num_layers, state_dim, bidirectional, token_ffn))
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList(
            self._stack(d_model, num_layers, state_dim, bidirectional, token_ffn))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_output = nn.Linear(d_model, 64)

    @staticmethod
    def _stack(d_model: int, num_layers: int, state_dim: int, bidirectional: bool,
               token_ffn: int) -> List[nn.Module]:
        block = BiMambaBlock if bidirectional else MambaBlock
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(block(d_model, state_dim))
            if token_ffn:
                layers.append(TokenFFN(d_model, token_ffn))
        return layers

    def encode_representation(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 2, 32, 32] -> [B, 32 antennas, 2 * 32]
        sequence = x.permute(0, 2, 1, 3).reshape(-1, 32, 64)
        sequence = self.encoder_input(sequence)
        for layer in self.encoder_layers:
            sequence = layer(sequence)
        sequence = self.encoder_norm(sequence)
        return sequence.reshape(sequence.shape[0], -1)

    def decode_representation(self, representation: torch.Tensor) -> torch.Tensor:
        sequence = representation.reshape(-1, 32, 64)
        for layer in self.decoder_layers:
            sequence = layer(sequence)
        output = self.decoder_output(self.decoder_norm(sequence))
        # [B, 32 antennas, 2 * 32] -> [B, 2, 32, 32]
        output = output.reshape(-1, 32, 2, 32).permute(0, 2, 1, 3)
        return torch.sigmoid(output) if self.output_activation == "sigmoid" else output
