from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class DTNN_layer(nn.Module):
    def __init__(self, input_node_dim, distance_dim, hidden_dim, output_node_dim):
        super(DTNN_layer, self).__init__()
        self.Wcf = nn.Linear(input_node_dim, hidden_dim)
        self.Wdf = nn.Linear(distance_dim, hidden_dim)
        self.Wfc = nn.Linear(hidden_dim, output_node_dim, bias=False)
        return

    def forward(self, x, distance):
        _, N, _ = x.size()
        x = x.unsqueeze(2).expand(-1, -1, N, -1)
        x = self.Wcf(x) * self.Wdf(distance)
        x = self.Wfc(x)
        x = torch.sum(x, dim=2)
        return x


class DeepTensorNeuralNetwork(nn.Module):
    def __init__(self, node_feature_dim, rbf_dim, hidden_dim, fc_hidden_dim, output_dim):
        super(DeepTensorNeuralNetwork, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.rbf_dim = rbf_dim
        self.distance_dim = rbf_dim

        self.gin_layers_dim = hidden_dim
        self.gin_layers_num = len(self.gin_layers_dim)
        self.DTNN_layers = nn.ModuleList()
        for layer_idx, (hidden_dim) in enumerate(self.gin_layers_dim):
            op = DTNN_layer(input_node_dim=node_feature_dim, distance_dim=self.distance_dim, hidden_dim=hidden_dim, output_node_dim=node_feature_dim)
            self.DTNN_layers.append(op)

        self.fc_hidden_dim = fc_hidden_dim
        self.output_dim = output_dim

        fc_layer_dim = [self.node_feature_dim] + self.fc_hidden_dim
        layers = OrderedDict()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(fc_layer_dim[:-1], fc_layer_dim[1:])):
            layers['fc layer {}'.format(layer_idx)] = nn.Linear(in_dim, out_dim)
        self.fc_represent_layers = nn.Sequential(layers)
        self.fc_layers = nn.Linear(fc_layer_dim[-1], self.output_dim)

        return

    def represent(self, x, distance):
        for layer_idx in range(self.gin_layers_num):
            x = self.DTNN_layers[layer_idx](x, distance)
            x = x + torch.tanh(x)
        x = torch.sum(x, dim=1)
        x = self.fc_represent_layers(x)
        return x

    def forward(self, x, distance):
        x = self.represent(x, distance)
        x = self.fc_layers(x)
        return x