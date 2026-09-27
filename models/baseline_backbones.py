"""Literature CSI-feedback codecs as backbones of the multi-link pipeline.

Each published codec is an encoder -> flatten(2048) -> FC(M) and FC(M) ->
reshape -> decoder autoencoder. In the multi-link pipeline the two FC layers
are replaced by the link projection (2048 -> 3 x 512) and the CPU aggregator
(1536 -> 2048), which are linear as well, so a backbone only has to provide

  encode_representation(x [B, 2, 32, 32]) -> [B, 2048]
  decode_representation(u [B, 2048])      -> [B, 2, 32, 32]

CRNet, CLNet and TransNet are built from the official PyTorch code cloned in
baselines/repos (see baselines/COMMITS.txt); CsiNet's official code is Keras,
so it is ported layer by layer below.
"""

import importlib.util
import logging
import os
import sys
import types
from contextlib import contextmanager

import torch
import torch.nn as nn

REPO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baselines', 'repos')
BACKBONES = ('csinet', 'crnet', 'clnet', 'transnet')


@contextmanager
def _stub_utils():
    """The cloned repos do `from utils import logger`; our project has its own `utils`."""
    saved = sys.modules.get('utils')
    stub = types.ModuleType('utils')
    stub.logger = logging.getLogger('baseline_repo')
    sys.modules['utils'] = stub
    try:
        yield
    finally:
        if saved is not None:
            sys.modules['utils'] = saved
        else:
            sys.modules.pop('utils', None)


def load_repo_module(repo: str, relpath: str, name: str):
    path = os.path.join(REPO_ROOT, repo, relpath)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; clone the baseline repos into {REPO_ROOT}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with _stub_utils():
        spec.loader.exec_module(module)
    return module


class _Backbone(nn.Module):
    """Common interface (matches MambaBackbone)."""
    feature_shape = (32, 64)

    @property
    def encoder_layers(self):
        return nn.ModuleList([self.encoder])

    @property
    def decoder_layers(self):
        return nn.ModuleList([self.decoder])


# ------------------------------------------------------------------- CsiNet
class _CsiNetRefine(nn.Module):
    """Residual block of the CsiNet decoder (Keras: conv8, conv16, conv2, add, LeakyReLU)."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU(0.3),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.3),
            nn.Conv2d(16, 2, 3, padding=1), nn.BatchNorm2d(2),
        )
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        return self.act(x + self.body(x))


class CsiNetBackbone(_Backbone):
    """Port of sydney222/Python_CsiNet (CsiNet_train.py, residual_num=2). Keras LeakyReLU uses alpha=0.3."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1), nn.BatchNorm2d(2), nn.LeakyReLU(0.3))
        self.decoder = nn.Sequential(_CsiNetRefine(), _CsiNetRefine(), nn.Conv2d(2, 2, 3, padding=1), nn.Sigmoid())

    def encode_representation(self, x):
        return self.encoder(x).flatten(1)

    def decode_representation(self, u):
        return self.decoder(u.view(-1, 2, 32, 32))


# -------------------------------------------------------------------- CRNet
class CRNetBackbone(_Backbone):
    """Official CRNet (Kylin9511/CRNet, models/crnet.py) without encoder_fc / decoder_fc."""

    def __init__(self):
        super().__init__()
        module = load_repo_module('CRNet', 'models/crnet.py', 'baseline_crnet')
        net = module.CRNet(reduction=4)
        self.encoder1, self.encoder2, self.encoder_conv = net.encoder1, net.encoder2, net.encoder_conv
        self.decoder_feature = net.decoder_feature
        self.encoder = nn.ModuleList([self.encoder1, self.encoder2, self.encoder_conv])
        self.decoder = self.decoder_feature

    def encode_representation(self, x):
        out = torch.cat((self.encoder1(x), self.encoder2(x)), dim=1)
        return self.encoder_conv(out).flatten(1)

    def decode_representation(self, u):
        return torch.sigmoid(self.decoder_feature(u.view(-1, 2, 32, 32)))


# -------------------------------------------------------------------- CLNet
class CLNetBackbone(_Backbone):
    """Official CLNet (SIJIEJI/CLNet, models/clnet.py) without replace_efc / replace_dfc."""

    def __init__(self):
        super().__init__()
        module = load_repo_module('CLNet', 'models/clnet.py', 'baseline_clnet')
        net = module.CLNet(reduction=4)
        enc, dec = net.encoder, net.decoder
        self.encoder1, self.sa, self.encoder2, self.se = enc.encoder1, enc.sa, enc.encoder2, enc.se
        self.encoder_conv = enc.encoder_conv
        self.decoder_feature, self.hsig = dec.decoder_feature, dec.hsig
        self.encoder = nn.ModuleList([self.encoder1, self.sa, self.encoder2, self.se, self.encoder_conv])
        self.decoder = self.decoder_feature

    def encode_representation(self, x):
        out = torch.cat((self.sa(self.encoder1(x)), self.se(self.encoder2(x))), dim=1)
        return self.encoder_conv(out).flatten(1)

    def decode_representation(self, u):
        return self.hsig(self.decoder_feature(u.view(-1, 2, 32, 32)))


# ----------------------------------------------------------------- TransNet
class TransNetBackbone(_Backbone):
    """Official TransNet (Treedy2020/TransNet, models/TransNet.py): d_model 64, 2 heads,
    2 (weight-shared) encoder and decoder layers, no dropout, linear output; without
    fc_encoder / fc_decoder.

    The released code feeds (B, 32, 64) tensors to attention layers built with
    batch_first=False, so attention runs across the samples of a batch. We use
    batch_first=True, i.e. attention over the 32 feature tokens of each sample,
    as the paper describes.
    """

    def __init__(self):
        super().__init__()
        module = load_repo_module('TransNet', 'models/TransNet.py', 'baseline_transnet')
        net = module.Transformer(d_model=64, num_encoder_layers=2, num_decoder_layers=2, nhead=2,
                                 reduction=4, dropout=0., batch_first=True)
        self.encoder, self.decoder = net.encoder, net.decoder

    def encode_representation(self, x):
        return self.encoder(x.reshape(-1, 32, 64)).flatten(1)

    def decode_representation(self, u):
        tokens = u.reshape(-1, 32, 64)
        return self.decoder(tokens, tokens).reshape(-1, 2, 32, 32)


def build_backbone(name: str) -> nn.Module:
    table = {'csinet': CsiNetBackbone, 'crnet': CRNetBackbone, 'clnet': CLNetBackbone,
             'transnet': TransNetBackbone}
    if name not in table:
        raise ValueError(f'unknown baseline backbone: {name}')
    return table[name]()
