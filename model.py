import torch.nn as nn
from layers import PhaseMask


class PhaseMaskModel(nn.Module):
    def __init__(self, shape, pad):

        super(PhaseMaskModel, self).__init__()
        self.pm = PhaseMask(shape, pad)

    def forward(self, x):
        return self.pm(x)
