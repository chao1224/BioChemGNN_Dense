from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolutionNetworkLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, batch_norm):
        super(GraphConvolutionNetworkLayer, self).__init__()
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


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation, batch_norm=True):
        super(GraphConvolutionNetwork, self).__init__()
        self.gcn_layers_dim = [node_feature_dim] + hidden_dim
        self.gcn_layers_num = len(self.gcn_layers_dim)
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.gcn_layers = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gcn_layers_dim[:-1], self.gcn_layers_dim[1:])):
            self.gcn_layers.append(GraphConvolutionNetworkLayer(in_dim, out_dim, activation, batch_norm))

        self.fc_layers = nn.Linear(self.gcn_layers_dim[-1], self.output_dim)
        return

    def represent(self, x, adjacency):
        for layer_idx in range(self.gcn_layers_num-1):
            x = x + torch.bmm(adjacency, x)
            x = self.gcn_layers[layer_idx](x)
            x_sum = torch.sum(x, dim=1)
            h = x_sum

        return h

    def forward(self, x, adjacency):
        x = self.represent(x, adjacency)
        x = self.fc_layers(x)
        return x