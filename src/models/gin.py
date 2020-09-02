from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class GraphIsomorphismNetworkLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, batch_norm):
        super(GraphIsomorphismNetworkLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = False

        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        return

    def forward(self, x):
        x = self.linear(x)
        B, N, d = x.size()
        x = x.reshape(B*N, d)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = x.reshape(B, N, d)
        if self.activation:
            x = self.activation(x)
        return x


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation, epsilon, batch_norm=True):
        super(GraphIsomorphismNetwork, self).__init__()
        self.gin_layers_dim = [node_feature_dim] + hidden_dim
        self.gin_layers_num = len(self.gin_layers_dim)
        self.epsilon = epsilon
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.gin_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gin_layers_dim[:-1], self.gin_layers_dim[1:])):
            self.gin_layers.append(GraphIsomorphismNetworkLayer(in_dim, out_dim, activation, batch_norm))

        self.fc_layers = nn.Linear(sum(self.gin_layers_dim), self.output_dim)
        return

    def represent(self, x, adjacency):
        h = []
        h.append(torch.sum(x, dim=1))
        for layer_idx in range(self.gin_layers_num-1):
            x = (1+self.epsilon) * x + torch.bmm(adjacency, x)
            # print(x.size())
            x = self.gin_layers[layer_idx](x)
            # print(x.size())
            x_sum = torch.sum(x, dim=1)
            h.append(x_sum)

        h = torch.cat(h, dim=1)

        return h

    def forward(self, x, adjacency):
        x = self.represent(x, adjacency)
        x = self.fc_layers(x)
        return x