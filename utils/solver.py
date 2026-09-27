import time
import os
import torch
from collections import namedtuple
from tensorboardX import SummaryWriter
from utils import logger
from utils.statics import AverageMeter, evaluator

__all__ = ['Trainer', 'Tester']


field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r""" The training pipeline for encoder-decoder architecture.

    Model selection uses the validation set only; the test set is never touched
    during training.

    With ``val_data`` the validation metric is the exact (over the 8 blockage
    patterns) expected NMSE on fixed random link states from the training
    distribution; otherwise the legacy single-state Tester is used.
    ``refine_start_epoch``: from this epoch on the model trains its closed-loop
    decoder (model.train_refine = True).
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', log_dir='./data_vision',
                 print_freq=20, val_freq=10, max_blockage=0.3, safe_link_prob=0.5,
                 model_args=None, val_data=None, val_states=None, refine_start_epoch=None):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.max_blockage = max_blockage
        self.safe_link_prob = safe_link_prob
        self.model_args = model_args
        self.val_data = val_data
        self.val_states = val_states
        self.refine_start_epoch = refine_start_epoch
        self.writer = SummaryWriter(log_dir=log_dir)

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.best_nmse = Result()

        self.tester = Tester(model, device, criterion, print_freq)

    def loop(self, epochs, train_loader, val_loader):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
        """

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep
            if hasattr(self.model, 'train_refine'):
                refine_on = (self.refine_start_epoch is not None and ep >= self.refine_start_epoch
                             and getattr(self.model, 'refine_steps', 0) > 0)
                if refine_on and not self.model.train_refine:
                    logger.info(f'=> epoch {ep}: closed-loop decoding switched on for training '
                                f'({self.model.refine_steps} steps)')
                self.model.train_refine = refine_on

            self.train_loss = self.train(train_loader)
            self.writer.add_scalar('train/loss', self.train_loss, global_step=ep)

            if ep % self.val_freq == 0 or ep == epochs:
                self.val_loss, _, nmse_db = self.val(val_loader)
                if self.val_loss is not None:
                    self.writer.add_scalar('val/loss', self.val_loss, global_step=ep)
                self.writer.add_scalar('val/nmse_db', nmse_db, global_step=ep)
            else:
                nmse_db = None

            # conduct saving, visualization and log printing
            self._loop_postprocessing(nmse_db)

    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        r""" Evaluate loss and NMSE on the validation set.

        Args:
            val_loader: (DataLoader): the validation data loader
        """

        self.model.eval()
        with torch.no_grad():
            if self.val_data is None:
                return self.tester(val_loader, verbose=False, label='Val')
            from evaluation.exact_eval import evaluate_states
            snr_db, blockage = self.val_states
            result = evaluate_states(self.model, self.val_data, snr_db, blockage, noise_seed=0)
            summary = result['summary']
            bands = ', '.join(f"{k}: {v['nmse_db']:.2f}" for k, v in summary['bands'].items()
                              if v['nmse_db'] is not None)
            logger.info(f"=> Val (exact, in-distribution, n={summary['n']}) "
                        f"NMSE dB: {summary['nmse_db']:.3f} | bands {bands}"
                        + (f" | modes {summary['selected_mode_share']}"
                           if 'selected_mode_share' in summary else ''))
            return None, None, summary['nmse_db']

    def _sample_link_state(self, batch_size):
        # Per sample and per link: SNR ~ U(0, 20) dB; the link is safe (p = 0)
        # with probability safe_link_prob, otherwise p ~ U(0, max_blockage).
        # The model draws the actual blockage events from these probabilities.
        shape = (batch_size, 3)
        snr_db = torch.rand(shape, device=self.device) * 20.0
        blockage_prob = torch.rand(shape, device=self.device) * self.max_blockage
        safe = torch.rand(shape, device=self.device) < self.safe_link_prob
        return snr_db, blockage_prob.masked_fill(safe, 0.0)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_route = AverageMeter('Iter route loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0].to(self.device)
            snr_db, blockage_prob = self._sample_link_state(sparse_gt.shape[0])
            output = self.model(sparse_gt, snr_db=snr_db,
                                blockage_prob=blockage_prob, hard_mask=False)
            loss = output['loss_total']

            # Scheduler update, backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Log update; .item() so the autograd graph is not kept alive.
            # The logged loss is the reconstruction MSE of the decoded estimate;
            # the router loss (select modes) is logged separately.
            iter_loss.update(output['loss_rec'].item())
            if 'loss_route' in output:
                iter_route.update(output['loss_route'].item())
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {lr:.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
                            + (f'route loss: {iter_route.avg:.3e} | ' if iter_route.count else '')
                            + f'time: {iter_time.avg:.3f}')

        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0],
                               global_step=self.cur_epoch)
        if iter_route.count:
            self.writer.add_scalar('train/route_loss', iter_route.avg, global_step=self.cur_epoch)
        logger.info(f'=> Train  Loss: {iter_loss.avg:.3e}'
                    + (f' | route loss: {iter_route.avg:.3e}' if iter_route.count else '') + '\n')

        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        r""" protected function which resume from checkpoint at the beginning of training.
        """

        if self.resume_file is None:
            return None
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}')
        with torch.serialization.safe_globals([Result]):
            checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_nmse = checkpoint.get('best_val_nmse', Result())
        self.cur_epoch += 1  # start from the next epoch

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                    f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, nmse_db):
        r""" private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_nmse': self.best_nmse,
            'model_args': self.model_args,
        }

        # save model with best validation NMSE
        if nmse_db is not None:
            if self.best_nmse.nmse is None or self.best_nmse.nmse > nmse_db:
                self.best_nmse = Result(nmse=nmse_db, epoch=self.cur_epoch)
                state['best_val_nmse'] = self.best_nmse
                self._save(state, name='best_val_nmse.pth')

        self._save(state, name='last.pth')

        # print current best results
        if self.best_nmse.nmse is not None:
            print(f'\n=! Best val NMSE dB: {self.best_nmse.nmse:.3e} '
                  f'(epoch={self.best_nmse.epoch})\n')


class Tester:
    r""" Evaluates loss, NMSE and (when raw frequency-domain CSI is available) rho.
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True, label='Test'):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader yielding (sparse,) or (sparse, raw).
        """

        self.model.eval()
        with torch.no_grad():
            loss, rho, nmse, nmse_db = self._iteration(test_data, label)
        if verbose:
            rho_text = f'{rho:.3e}' if rho is not None else 'n/a'
            print(f'\n=> {label} result: \nloss: {loss:.3e}'
                  f'    rho: {rho_text}    NMSE: {nmse:.3e}'
                  f'    NMSE dB: {nmse_db:.3e}\n')
        return loss, rho, nmse_db

    def _iteration(self, data_loader, label='Test'):
        r""" protected function which test the model on given data loader for one epoch.
        """

        iter_rho = AverageMeter('Iter rho')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        nmse_sum = 0.0
        sample_count = 0
        has_rho = False
        time_tmp = time.time()

        for batch_idx, batch in enumerate(data_loader):
            sparse_gt = batch[0].to(self.device)
            raw_gt = batch[1].to(self.device) if len(batch) > 1 else None
            output = self.model(sparse_gt, hard_mask=True)
            sparse_pred = output['reconstruction'] if isinstance(output, dict) else output
            loss = self.criterion(sparse_pred, sparse_gt)
            rho, nmse, _ = evaluator(sparse_pred, sparse_gt, raw_gt)

            batch_size = sparse_gt.shape[0]
            iter_loss.update(loss.item(), batch_size)
            if rho is not None:
                has_rho = True
                iter_rho.update(rho.item(), batch_size)
            nmse_sum += nmse.item() * batch_size
            sample_count += batch_size
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'NMSE: {nmse_sum / sample_count:.3e} | '
                            f'time: {iter_time.avg:.3f}')

        nmse_global = nmse_sum / max(sample_count, 1)
        nmse_db_global = 10.0 * torch.log10(
            torch.as_tensor(nmse_global).clamp_min(torch.finfo(torch.float32).eps)
        ).item()
        rho_avg = iter_rho.avg if has_rho else None
        rho_text = f'{rho_avg:.3e}' if rho_avg is not None else 'n/a'
        logger.info(f'=> {label} loss: {iter_loss.avg:.3e}  rho: {rho_text}  '
                    f'NMSE: {nmse_global:.3e}  NMSE dB: {nmse_db_global:.3e}\n')

        return iter_loss.avg, rho_avg, nmse_global, nmse_db_global
