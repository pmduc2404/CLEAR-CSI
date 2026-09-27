import argparse

from utils.model_args import add_model_args

parser = argparse.ArgumentParser(description='CSIMamba multi-link CSI feedback training')


# ========================== Indispensable arguments ==========================

parser.add_argument('--data-dir', type=str, required=True,
                    help='the path of dataset.')
parser.add_argument('--scenario', type=str, required=True, choices=["in", "out"],
                    help="the channel scenario")
parser.add_argument('-b', '--batch-size', type=int, required=True, metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, metavar='N', required=True,
                    help='number of data loading workers')


# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='legacy evaluation on the test set at --snr-db / --blockage-prob')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')
parser.add_argument('--output-dir', type=str, default='./evaluation_results')

# Optimisation
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('--lr', type=float, default=None,
                    help='peak learning rate (default: 1e-4 const, 2e-4 cosine)')
parser.add_argument('--warmup-epochs', type=int, default=5,
                    help='cosine: linear warmup length in epochs')
parser.add_argument('--eta-min', type=float, default=1e-6,
                    help='cosine: final learning rate')
parser.add_argument('--adam-eps', type=float, default=1e-8,
                    help='Adam epsilon (acts as an implicit LR cap for very small gradients)')

# Training link-state distribution
parser.add_argument('--max-blockage', type=float, default=0.3,
                    help='training draws a risky link\'s blockage probability from U(0, max)')
parser.add_argument('--safe-link-prob', type=float, default=0.5,
                    help='training marks each link safe (blockage probability 0) with this probability')

# Closed loop and validation
parser.add_argument('--refine-start-frac', type=float, default=0.8,
                    help='closed-loop decoding (--refine-steps) is trained from this fraction of the epochs on')
parser.add_argument('--val-freq', type=int, default=10,
                    help='validate every N epochs')
parser.add_argument('--val-samples', type=int, default=2000,
                    help='validation samples for the exact in-distribution metric (0 = legacy validation)')
parser.add_argument('--val-seed', type=int, default=1234,
                    help='seed of the fixed validation link states')

add_model_args(parser)

args = parser.parse_args()
