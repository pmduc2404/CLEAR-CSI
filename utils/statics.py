import torch

__all__ = ['AverageMeter', 'evaluator']


class AverageMeter(object):

    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(sparse_pred, sparse_gt, raw_gt):
   
    with torch.no_grad():
        # Basic params
        nt = 32
        nc = 32
        nc_expand = 257

        # De-centralize
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5

        # Calculate linear NMSE first; convert to dB only for reporting.
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        power = power_gt.sum(dim=[1, 2]).clamp_min(torch.finfo(sparse_gt.dtype).eps)
        nmse = (mse.sum(dim=[1, 2]) / power).mean()
        nmse_db = 10 * torch.log10(nmse.clamp_min(torch.finfo(nmse.dtype).eps))

        # Rho needs the raw frequency-domain CSI, which only the test set has
        if raw_gt is None:
            return None, nmse, nmse_db

        # Calculate the Rho
        n = sparse_pred.size(0)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last
        zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
        sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
        sparse_pred = torch.view_as_complex(sparse_pred.contiguous())
        raw_pred = torch.fft.fft(sparse_pred, dim=2)
        raw_pred = torch.view_as_real(raw_pred)[:, :, :125, :]
        
        norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        denominator = (norm_pred * norm_gt).clamp_min(torch.finfo(norm_pred.dtype).eps)
        rho = (norm_cross / denominator).real.mean()

        return rho, nmse, nmse_db
