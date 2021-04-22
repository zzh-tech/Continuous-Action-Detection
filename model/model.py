import torch.nn as nn
from importlib import import_module


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        model_name = para.model
        module = import_module('model.' + model_name)
        self.model = module.Model(para)

    def __repr__(self):
        return self.model.__repr__()

    def forward(self, x):
        y = self.model(x)

        return y
