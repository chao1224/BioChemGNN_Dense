from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class NeuralFingerprint(nn.Module):
    def __init__(self, node_feature_dim, nef_fp_hidden_dim, nef_fp_length, fc_hidden_dim, output_dim):
        super(NeuralFingerprint, self).__init__()
        self.nef_fp_length = nef_fp_length
        self.nef_layers_dim = [node_feature_dim] + nef_fp_hidden_dim
        self.nef_layers_num = len(self.nef_layers_dim)

        self.nef_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.nef_layers_dim[:-1], self.nef_layers_dim[1:])):
            op = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.nef_layers.append(op)
        self.nef_mapping = nn.ModuleList()
        for layer_dim in self.nef_layers_dim:
            op = nn.Sequential(nn.Linear(layer_dim, self.nef_fp_length), nn.Softmax())
            self.nef_mapping.append(op)

        self.fc_hidden_dim = fc_hidden_dim
        self.output_dim = output_dim

        layer_dim = [self.nef_fp_length] + self.fc_hidden_dim
        layers = OrderedDict()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(layer_dim[:-1], layer_dim[1:])):
            layers['fc layer {}'.format(layer_idx)] = nn.Linear(in_dim, out_dim)
            layers['relu {}'.format(layer_idx)] = nn.ReLU()
        self.fc_represent_layers = nn.Sequential(layers)
        self.fc_layers = nn.Linear(layer_dim[-1], self.output_dim)
        return

    def represent(self, x, adjacency):
        h = []
        h.append(self.nef_mapping[0](x))
        for layer_idx in range(self.nef_layers_num-1):
            x = x + torch.bmm(adjacency, x)
            x = self.nef_layers[layer_idx](x)
            h.append(self.nef_mapping[layer_idx+1](x))

        h = torch.stack(h, dim=0)
        h = torch.sum(h, dim=2)
        h = torch.sum(h, dim=0)
        h = self.fc_represent_layers(h)

        return h

    def forward(self, x, adjacency):
        x = self.represent(x, adjacency)
        x = self.fc_layers(x)
        return x