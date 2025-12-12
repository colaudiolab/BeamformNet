import lap
import numpy as np
import torch
from torch import nn

from utils.metrics import RMSE_mode


class AsymmetricLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.gamma_neg = 4
        self.gamma_pos = 1
        self.clip = 0.05
        self.disable_torch_grad_focal_loss = True
        self.eps = 1e-8
        self.use_sigmoid = False

    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes)
        labels: (batch_size, num_classes) 01矩阵
        """
        y = labels

        # Calculating Probabilities
        if self.use_sigmoid:
            x_sigmoid = torch.sigmoid(logits)
        else:
            x_sigmoid = logits
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (probability shifting)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(dim=1).mean()


class RMSPELoss(nn.Module):
    def __init__(self, mod,deg_l,deg_u,deg_p):
        super().__init__()

        self.mod = mod
        self.deg_l = deg_l
        self.deg_u = deg_u
        self.deg_p = deg_p

    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes)
        labels: (batch_size, num_classes) 01矩阵
        """
        # logits_numpy = logits.detach().cpu().numpy()
        labels_numpy = labels.detach().cpu().numpy()
        bs = labels_numpy.shape[0]
        loss_list = torch.zeros(bs).to(labels.device)
        for bs_index in range(bs):
            label = (np.where(labels_numpy[bs_index] == 1)[0] * self.deg_p - self.deg_u) * np.pi / 180
            d = label.shape[0]
            doa = logits[bs_index][:d].detach().cpu().numpy()

            m = len(doa)
            n = len(label)
            min_m_n = min(m, n)

            costs_matrix = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    costs_matrix[i, j] = RMSE_mode(doa[i], label[j])

            cor_indexs = lap.lapjv(costs_matrix, extend_cost=True)[1]

            loss = 0
            for row_index, col_index in enumerate(cor_indexs):
                loss += (((logits[bs_index][row_index] - label[col_index]) + self.mod / 2) % self.mod - self.mod / 2)**2

            rmse_loss = torch.sqrt(loss / min_m_n)
            loss_list[bs_index] = rmse_loss

        return loss_list.mean()