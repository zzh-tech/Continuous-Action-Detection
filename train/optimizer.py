from importlib import import_module
import torch.optim.lr_scheduler as lr_scheduler


class Optimizer:
    def __init__(self, para, target):
        # create optimizer
        # trainable = filter(lambda x: x.requires_grad, target.parameters())
        trainable = target.parameters()
        optimizer_name = para.optimizer
        lr = para.lr
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        # create scheduler
        milestones = para.milestones
        gamma = para.decay_gamma
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()
