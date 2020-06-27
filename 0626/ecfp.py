from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class ECFPNetwork(nn.Module):
    def __init__(self, ECFP_dim, hidden_dim, output_dim):
        super(ECFPNetwork, self).__init__()
        self.ECFP_dim = ECFP_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layer_dim = [self.ECFP_dim] + self.hidden_dim + [self.output_dim]
        layer_num = len(layer_dim) - 1

        fc_layers = OrderedDict()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(layer_dim[:-1], layer_dim[1:])):
            fc_layers['fc layer {}'.format(layer_idx)] = nn.Linear(in_dim, out_dim)
            if layer_idx + 1 < layer_num:
                fc_layers['relu {}'.format(layer_idx)] = nn.ReLU()
        self.fc = nn.Sequential(fc_layers)
        return

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_representation(self, x):
        x = self.fc(x)
        return x