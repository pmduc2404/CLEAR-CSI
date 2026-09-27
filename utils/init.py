import os
import random
import numpy as np
import torch

from utils import logger, line_seg

__all__ = ["init_device", "init_model", "build_model", "load_checkpoint", "load_model_from_checkpoint"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if seed is None:
            torch.backends.cudnn.benchmark = True
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def build_model(kwargs):
    from models import MambaMultiLinkCSI
    from utils.model_args import drop_removed_kwargs
    kwargs = drop_removed_kwargs(kwargs)
    if kwargs['mode'] not in MambaMultiLinkCSI.MODES:
        raise ValueError(f"unknown model: {kwargs['mode']}")
    return MambaMultiLinkCSI(**kwargs)


def load_checkpoint(path):
    from utils.solver import Result
    with torch.serialization.safe_globals([Result]):
        return torch.load(path, map_location=torch.device('cpu'))


def load_model_from_checkpoint(path, overrides=None, fallback_kwargs=None, strict=True):
    """Build the model a checkpoint was trained with and load its weights.

    New checkpoints store their MambaMultiLinkCSI kwargs under 'model_args';
    for older ones pass ``fallback_kwargs``. ``overrides`` (e.g. refine_steps)
    are applied on top.
    """
    checkpoint = load_checkpoint(path)
    kwargs = dict(checkpoint.get('model_args') or fallback_kwargs or {})
    if not kwargs:
        raise ValueError(f'{path} has no model_args; pass fallback_kwargs')
    from utils.model_args import drop_removed_kwargs
    kwargs = drop_removed_kwargs(kwargs)
    kwargs.update(overrides or {})
    model = build_model(kwargs)
    model.load_state_dict(checkpoint['state_dict'], strict=strict)
    return model, checkpoint, kwargs


def init_model(args):
    from utils.model_args import model_kwargs, structural_mismatch

    kwargs = model_kwargs(args)
    model = build_model(kwargs)
    model_name = kwargs['mode']

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        checkpoint = load_checkpoint(args.pretrained)
        stored = checkpoint.get('model_args')
        if stored:
            mismatch = structural_mismatch(stored, kwargs)
            if mismatch:
                logger.warning(f'pretrained checkpoint was trained with different settings: {mismatch}')
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    flops, params = 'not profiled', f'{sum(p.numel() for p in model.parameters()):,}'

    # Model info logging
    logger.info(f'=> Model Name: {model_name} [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: {kwargs}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    model.model_kwargs = kwargs
    return model
