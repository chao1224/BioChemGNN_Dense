from collections import *
import torch
from torch import nn
from torch.nn import functional as F


class Set2Set(torch.nn.Module):
    def __init__(self, input_dim, processing_steps, num_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, self.num_layers)
        return

    def forward(self, x_represent, node_masking):
        batch_size, n = x_represent.size()[0], x_represent.size()[1]

        h = (torch.zeros(self.num_layers, batch_size, self.input_dim).to(device=x_represent.device),
             torch.zeros(self.num_layers, batch_size, self.input_dim).to(device=x_represent.device))
        q_star = torch.zeros(1, batch_size, self.output_dim).to(device=x_represent.device)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star, h)
            e = torch.bmm(x_represent, q.squeeze(0).unsqueeze(2))
            e = e.masked_fill(node_masking == 0, float('-inf'))
            a = nn.Softmax(dim=1)(e)
            r = torch.sum(a * x_represent * node_masking, dim=1).unsqueeze(0)
            q_star = torch.cat([q, r], dim=2)

        q_star = q_star.squeeze(0)
        return q_star


class EdegeNeuralNetwork_MessagePassingLayer(torch.nn.Module):
    def __init__(self, node_hidden_dim, fc_dim, edge_feature_dim):
        super(EdegeNeuralNetwork_MessagePassingLayer, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        fc_dim = [edge_feature_dim] + fc_dim + [self.node_hidden_dim * self.node_hidden_dim]

        layers = OrderedDict()
        layers_num = len(fc_dim) - 1
        for layer_idx, (in_dim, out_dim) in enumerate(zip(fc_dim[:-1], fc_dim[1:])):
            layers['edge projection fc layer {}'.format(layer_idx)] = nn.Linear(in_dim, out_dim)
            if layer_idx + 1 < layers_num:
                layers['edge projection activateion {}'.format(layer_idx)] = nn.ReLU()
        self.edge_fc_layers = nn.Sequential(layers)

        return

    def forward(self, N, x, edge):
        BN2, _ = edge.size()
        e = self.edge_fc_layers(edge)
        e = e.reshape(BN2, self.node_hidden_dim, self.node_hidden_dim)
        # x = x.unsqueeze(2).expand(-1, -1, N).reshape(-1, self.node_hidden_dim)
        x = x.unsqueeze(1).expand(-1, N, -1).reshape(-1, self.node_hidden_dim)
        x = x.unsqueeze(2)
        e = torch.bmm(e, x).squeeze()
        return e


class EdgeNeuralNetwork(torch.nn.Module):
    def __init__(
            self, node_feature_dim, edge_feature_dim,
            hidden_dim, fc_dim, gru_layer_num, enn_layer_num,
            readout_func, set2set_processing_steps, set2set_num_layers,
            output_dim
    ):
        super(EdgeNeuralNetwork, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.gru_layer_num = gru_layer_num
        self.enn_layer_num = enn_layer_num
        self.output_dim = output_dim

        self.embedding_layer = nn.Linear(self.node_feature_dim, self.hidden_dim)

        self.enn_layer = EdegeNeuralNetwork_MessagePassingLayer(node_hidden_dim=hidden_dim, fc_dim=self.fc_dim, edge_feature_dim=edge_feature_dim)
        self.gru_layer = nn.GRU(self.hidden_dim, self.hidden_dim, self.gru_layer_num)

        self.readout_func = readout_func
        if self.readout_func == 'set2set':
            self.set2set = Set2Set(input_dim=self.hidden_dim, processing_steps=set2set_processing_steps, num_layers=set2set_num_layers)
            self.fc_layer = nn.Linear(self.hidden_dim*2, self.output_dim)
        elif self.readout_func == 'sum':
            self.fc_layer = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            raise ValueError('Read-out function {} not included.'.format(self.readout_func))

        return

    def represent(self, x, edge, adjacency):
        message_masking = (torch.sum(x, 2).unsqueeze(2).expand(-1, -1, self.hidden_dim) > 0).type_as(x)
        x = self.embedding_layer(x)
        hidden_layers = [x]
        B, N, node_dim = x.size()

        adjacency = adjacency.unsqueeze(3).expand(-1, -1, -1, self.hidden_dim).reshape(B*N*N, self.hidden_dim)
        edge = edge.reshape(B*N*N, -1)

        for _ in range(self.enn_layer_num):
            h = hidden_layers[-1].reshape(B*N, -1)
            message = self.enn_layer(N, h, edge)

            message = adjacency * message
            message = message.reshape(B, N, N, -1)

            message = message.sum(dim=1).reshape(-1, self.hidden_dim).unsqueeze(0)
            x = hidden_layers[-1].reshape(-1, self.hidden_dim).unsqueeze(0)

            h, _ = self.gru_layer(message, x)
            h = h.squeeze().reshape(B, N, self.hidden_dim)

            h = message_masking * h

            hidden_layers.append(h)

        return hidden_layers[-1]

    def forward(self, x, edge, adjacency):
        node_masking = (torch.sum(x, 2) > 0).type_as(x).unsqueeze(2)

        x = self.represent(x, edge, adjacency)
        if self.readout_func == 'set2set':
            x = self.set2set(x, node_masking)
        elif self.readout_func == 'sum':
            x = torch.sum(x, dim=1)
        x = self.fc_layer(x)
        return x


