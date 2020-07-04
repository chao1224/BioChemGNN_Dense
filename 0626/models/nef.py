from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class NeuralFingerprint(nn.Module):
    def __init__(self, node_feature_dim, ecfp_dim, hidden_dim, output_dim):
        super(ECFPNetwork, self).__init__()
        self.fp_length = 50
        self.nef_layers = [node_feature_dim] + [20, 20, 20, 20]
        self.hidden_dim = [100]
        self.output_dim = output_dim

        layer_dim = [self.ECFP_dim] + self.hidden_dim

        self.nef_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(nef_layers[:-1], nef_layers[1:])):
            op = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.nef_layers.append(op)
        self.nef_mapping = nn.ModuleList()
        for layer_dim in nef_layers:
            self.nef_mapping.append(nn.Linear(layer_dim, self.fp_length))

        layers = OrderedDict()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(layer_dim[:-1], layer_dim[1:])):
            layers['fc layer {}'.format(layer_idx)] = nn.Linear(in_dim, out_dim)
            layers['relu {}'.format(layer_idx)] = nn.ReLU()
        self.represent_layers = nn.Sequential(layers)
        self.fc_layers = nn.Linear(layer_dim[-1], self.output_dim)
        return

    def represent(self, x):
        return x

    def forward(self, x, adjacency):
        h.append(x)
        # for i in range(self.ney_layer):

        x = self.fc_layers(x)
        return x