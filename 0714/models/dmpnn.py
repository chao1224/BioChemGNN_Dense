from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class DirectedMessagePassingNeuralNetwork(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, layer_size, output_dim):
        super(DirectedMessagePassingNeuralNetwork, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_size = layer_size
        self.layer_size = 1

        self.activation = nn.ReLU()

        self.W_input = nn.Linear(node_feature_dim+edge_feature_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_output = nn.Linear(node_feature_dim+hidden_dim, hidden_dim)
        self.fc_layers = nn.Linear(hidden_dim, self.output_dim)

        return

    def represent(self, x, edge, adjacency):
        B, N, d = x.size()
        x_input = x
        x = x.unsqueeze(1).expand(-1, N, -1, -1)
        h = self.W_input(torch.cat([x, edge], dim=-1)) * adjacency.unsqueeze(3)
        h_0 = self.activation(h)

        hidden_layers = [h_0]

        for i in range(self.layer_size):
            h = hidden_layers[-1]
            m = torch.sum(h, dim=2).unsqueeze(1).expand(-1, N, -1, -1) - h
            h = self.activation(h_0 + self.W_h(m)) * adjacency.unsqueeze(3)
            hidden_layers.append(h)

        h = hidden_layers[-1]
        M = torch.sum(h, dim=2)
        h = self.W_output(torch.cat([x_input, M], dim=-1)) * (adjacency.sum(dim=2)>0).unsqueeze(2)
        h = self.activation(h)
        return h

    def forward(self, x, edge, adjacency):
        x = self.represent(x, edge, adjacency)
        # print('node repr\t', x.size())
        h_v = torch.sum(x, dim=1)
        h_v = self.fc_layers(h_v)
        # print('graph repr\t', h_v.size())
        return h_v


