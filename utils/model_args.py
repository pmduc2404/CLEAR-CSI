"""Model hyper-parameters shared by main.py and the analysis scripts.

The defaults here are the CLEAR configuration (transmit power normalization,
outage-factorized router).
"""

import argparse

MODES = ['equal_3link', 'fixed_3link', 'greedy_capacity', 'proportional_capacity',
         'multi_link', 'multi_link_state', 'adaptive_wo_snr', 'risk_switch',
         'mode_select', 'mode_select_content']

# MambaMultiLinkCSI keyword -> argparse attribute
_KWARG_TO_ATTR = {
    'd_model': 'd_model', 'mode': 'model', 'snr_db': 'snr_db', 'blockage_prob': 'blockage_prob',
    'feedback_mode': 'feedback_mode', 'fixed_allocation': 'fixed_allocation',
    'total_budget': 'total_budget',
    'state_conditioning': 'state_conditioning', 'capacity_prior': 'capacity_prior',
    'risk_threshold': 'risk_threshold', 'select_floor': 'select_floor',
    'tx_norm': 'tx_norm', 'io_standardize': 'io_standardize',
    'bidirectional': 'bidirectional', 'token_ffn': 'token_ffn',
    'cpu_extra_layers': 'cpu_extra_layers', 'router': 'router',
    'refine_steps': 'refine_steps', 'refine_step': 'refine_step',
    'refine_refiner': 'refine_refiner', 'refine_forward': 'refine_forward',
    'refine_detach_forward': 'refine_detach_forward',
    'backbone': 'backbone',
}

# Settings that change the parameter layout (a checkpoint only loads if they match).
STRUCTURAL_KWARGS = ('d_model', 'mode', 'state_conditioning', 'capacity_prior',
                     'io_standardize', 'bidirectional', 'token_ffn',
                     'cpu_extra_layers', 'router', 'refine_step', 'refine_refiner',
                     'refine_forward', 'backbone')

# Values assumed for keys missing from older checkpoints' model_args.
KWARG_DEFAULTS = {'backbone': 'mamba'}

# Settings of an earlier model version that older checkpoints still store.
REMOVED_KWARGS = ('num_experts', 'bottleneck')


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-d', '--d_model', type=int, default=64, metavar='N',
                       help='Mamba feature dimension.')
    group.add_argument('--model', type=str, default='multi_link', choices=MODES)
    group.add_argument('--snr-db', type=float, nargs=3, default=[20.0, 10.0, 3.0],
                       metavar=('SNR1', 'SNR2', 'SNR3'),
                       help='uplink SNRs (dB) used for legacy validation and evaluation')
    group.add_argument('--blockage-prob', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       metavar=('P1', 'P2', 'P3'),
                       help='uplink blockage probabilities used for legacy validation and evaluation')
    group.add_argument('--feedback-mode', choices=['ideal', 'noisy'], default='noisy')
    group.add_argument('--fixed-allocation', type=float, nargs=3, default=[0.5, 0.3, 0.2],
                       metavar=('A1', 'A2', 'A3'))
    group.add_argument('--total-budget', type=int, default=512,
                       help='total real channel uses K over the three uplinks')
    group.add_argument('--no-state-conditioning', dest='state_conditioning', action='store_false',
                       help='ablation: encoder FiLM and CPU fusion do not see the link state')
    group.add_argument('--no-capacity-prior', dest='capacity_prior', action='store_false',
                       help='ablation: learned allocators start from scratch, not the capacity split')
    group.add_argument('--risk-threshold', type=float, default=0.02,
                       help='risk_switch: links with blockage probability below this are safe')
    group.add_argument('--select-floor', type=float, default=1.0,
                       help='mode_select: share of the reconstruction loss spread evenly over all candidates')
    # CLEAR options
    group.add_argument('--tx-norm', action=argparse.BooleanOptionalAction, default=True,
                       help='physical transmit power normalization (--no-tx-norm = legacy leaky channel)')
    group.add_argument('--io-standardize', action='store_true',
                       help='standardize the input with training statistics; linear output')
    group.add_argument('--bidirectional', action='store_true', help='bidirectional Mamba backbone')
    group.add_argument('--token-ffn', type=int, default=0,
                       help='hidden size of a per-token FFN after each Mamba block (0 = off)')
    group.add_argument('--cpu-extra-layers', type=int, default=0,
                       help='extra residual dense blocks after CPU fusion (closed-loop control; 3 ~ 25M MAC)')
    group.add_argument('--router', choices=['regression', 'ofdr'], default='ofdr',
                       help='router of the select modes (ofdr = outage-factorized, risk-priced)')
    group.add_argument('--refine-steps', type=int, default=0,
                       help='closed-loop CPU decoding steps (evaluation, and training after --refine-start-frac)')
    group.add_argument('--refine-step', choices=['rule', 'learned'], default='rule',
                       help='per-link closed-loop step: fixed SNR rule or learned around it')
    group.add_argument('--refine-refiner', action='store_true',
                       help='add the learned zero-initialised correction R(u, b) to the closed loop')
    group.add_argument('--refine-forward', choices=['tied', 'untied', 'none'], default='tied',
                       help='closed-loop forward operator: the UE chain (tied), a trained copy '
                            '(untied, control), or no re-encoding (none, control; needs --refine-refiner)')
    group.add_argument('--refine-detach-forward', action='store_true',
                       help='ablation: no gradient through the re-encoded latent')
    group.add_argument('--backbone', choices=['mamba', 'csinet', 'crnet', 'clnet', 'transnet'],
                       default='mamba',
                       help='codec backbone; the literature codecs come from baselines/repos')
    return parser


def model_kwargs(args):
    """argparse namespace -> MambaMultiLinkCSI keyword arguments."""
    kwargs = {kwarg: getattr(args, attr) for kwarg, attr in _KWARG_TO_ATTR.items()}
    kwargs['snr_db'] = list(kwargs['snr_db'])
    kwargs['blockage_prob'] = list(kwargs['blockage_prob'])
    kwargs['fixed_allocation'] = list(kwargs['fixed_allocation'])
    if kwargs['mode'] == 'mode_select_content':
        kwargs['router'] = 'regression'
    return kwargs


def drop_removed_kwargs(kwargs):
    """Stored model_args -> constructor kwargs: drop settings of an earlier model version."""
    if kwargs.get('bottleneck', 'none') != 'none':
        raise ValueError('this checkpoint uses a bottleneck block that is no longer supported')
    return {key: value for key, value in kwargs.items() if key not in REMOVED_KWARGS}


def structural_mismatch(a, b):
    """Structural keys whose values differ between two kwargs dicts."""
    def value(kwargs, key):
        return kwargs.get(key, KWARG_DEFAULTS.get(key))
    return {key: (value(a, key), value(b, key)) for key in STRUCTURAL_KWARGS
            if value(a, key) != value(b, key)}
