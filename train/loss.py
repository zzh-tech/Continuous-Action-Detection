import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def MSE():
    return nn.MSELoss()


def L1():
    return nn.L1Loss()


class CrossEntropy(_Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        if len(x.shape) == 3:
            return self.loss(x, y)
        elif len(x.shape) == 4:
            N, C, T, L = x.shape
            y = y.unsqueeze(-1).expand(-1, -1, L)
            return self.loss(x, y)


# mixed loss
# cross entropy loss & smoothing loss
class Mixed(_Loss):
    def __init__(self):
        super(Mixed, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduction='none')  # keep the size

    def forward(self, x, y):
        r1 = 0.45
        r2 = 8
        if len(x.shape) == 3:
            loss1 = self.cross_entropy(x, y)
            loss2 = self.mse(F.log_softmax(x[:, :, 1:], dim=1), F.log_softmax(x.detach()[:, :, :-1], dim=1))
            loss2 = r1 * torch.mean(torch.clamp(loss2, min=0, max=r2 ** 2))
            loss = loss1 + loss2
            return loss
        elif len(x.shape) == 4:
            N, C, T, L = x.shape
            y = y.unsqueeze(-1).expand(-1, -1, L)
            loss1 = self.cross_entropy(x, y)
            loss2 = self.mse(F.log_softmax(x[:, :, 1:, :], dim=1), F.log_softmax(x.detach()[:, :, :-1, :], dim=1))
            loss2 = r1 * torch.mean(torch.clamp(loss2, min=0, max=r2 ** 2))
            loss = loss1 + loss2
            return loss
