import json
import math
import os
import torch
import torch.nn as nn
from utils.parser import args
from utils import logger, Trainer
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataloader import Cost2100DataLoader
from evaluation import evaluate_model


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    data = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario,
        temporal=False)
    train_loader, val_loader, test_loader = data()

    # Define model
    model = init_model(args)
    if getattr(model, 'io_standardize', False) and args.pretrained is None and args.resume is None:
        train_x = data.train_dataset.tensors[0]
        mean = train_x.mean(dim=0)
        std = (train_x - mean).std()
        model.set_io_statistics(mean, std)
        logger.info(f'=> input standardization: global std {float(std):.4f}')
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Legacy inference mode (single fixed state, test set)
    if args.evaluate:
        summary = evaluate_model(model, test_loader, device, args.output_dir,
                                 snr_db=args.snr_db,
                                 blockage_prob=args.blockage_prob)
        print(f'Evaluation: {summary}')
        return

    # Define optimizer and scheduler
    lr_init = args.lr if args.lr is not None else (1e-4 if args.scheduler == 'const' else 2e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr_init, eps=args.adam_eps)

    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=args.warmup_epochs * len(train_loader),
                                            eta_min=args.eta_min)

    # Validation: exact expected NMSE on fixed link states from the training distribution
    val_data, val_states = None, None
    if args.val_samples > 0:
        from evaluation.exact_eval import sample_states
        val_data = data.val_dataset.tensors[0][:args.val_samples]
        val_states = sample_states(val_data.shape[0], args.val_seed, 'train',
                                   args.max_blockage, args.safe_link_prob, args.risk_threshold)
    refine_start_epoch = None
    if args.refine_steps > 0:
        refine_start_epoch = max(1, int(math.floor(args.refine_start_frac * args.epochs)) + 1)

    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume,
                      save_path=os.path.join(args.output_dir, 'checkpoints'),
                      log_dir=os.path.join(args.output_dir, 'tensorboard'),
                      val_freq=args.val_freq,
                      max_blockage=args.max_blockage,
                      safe_link_prob=args.safe_link_prob,
                      model_args=model.model_kwargs,
                      val_data=val_data,
                      val_states=val_states,
                      refine_start_epoch=refine_start_epoch)

    # Start training (model selection on the validation set only)
    trainer.loop(args.epochs, train_loader, val_loader)

    # Final summary on validation only; the test set is evaluated once, later,
    # with analysis/eval_exact.py --split test on the final models.
    if val_data is not None:
        from evaluation.exact_eval import evaluate_states
        result = evaluate_states(model, val_data, *val_states, noise_seed=0)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'val_summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(result['summary'], handle, indent=2)
        print(f"\n=! Final validation (exact, in-distribution): {result['summary']['nmse_db']:.3f} dB\n")


if __name__ == "__main__":
    main()
