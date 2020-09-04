from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation, epsilon):
        super(GraphIsomorphismNetwork, self).__init__()
        self.gin_layers_dim = [node_feature_dim] + hidden_dim
        self.gin_layers_num = len(self.gin_layers_dim)
        self.epsilon = epsilon
        self.output_dim = output_dim
        print(self.gin_layers_dim)
        print(sum(self.gin_layers_dim))

        if activation is not None:
            activation = getattr(F, activation)

        self.gin_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gin_layers_dim[:-1], self.gin_layers_dim[1:])):
            if activation is None:
                op = nn.Sequential(nn.Linear(in_dim, out_dim))
            else:
                op = nn.Sequential(nn.Linear(in_dim, out_dim), activation)
            self.gin_layers.append(op)

        self.fc_layers = nn.Linear(sum(self.gin_layers_dim), self.output_dim)
        return

    def represent(self, x, adjacency):
        h = []
        h.append(torch.sum(x, dim=1))
        for layer_idx in range(self.gin_layers_num-1):
            x = (1+self.epsilon) * x + torch.bmm(adjacency, x)
            x = self.gin_layers[layer_idx](x)
            x_sum = torch.sum(x, dim=1)
            h.append(x_sum)

        h = torch.cat(h, dim=1)

        return h

    def forward(self, x, adjacency):
        x = self.represent(x, adjacency)
        x = self.fc_layers(x)
        return x