import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_RBF_dimension(low=0., high=30., gap=0.1):
    num_centers = int(np.ceil((high - low) / gap))
    return num_centers


def RBFExpansion(distance_matrix, low=0., high=30., gap=0.1, gamma=10.):
    num_centers = int(np.ceil((high - low) / gap))
    centers = np.linspace(low, high, num_centers)
    centers = torch.as_tensor(centers, device=distance_matrix.device).float()

    radial = distance_matrix - centers
    x = torch.exp(-gamma * (radial ** 2))
    return x


class ShiftedSoftPlus(nn.Module):
    def __init__(self):
        super(ShiftedSoftPlus, self).__init__()
        return

    def forward(self, x):
        return F.softplus(x) - np.log(2.0)


class CFConv(nn.Module):
    def __init__(self, rbf_dim, node_dim):
        super(CFConv, self).__init__()
        self.rbf_dim = rbf_dim
        self.node_dim = node_dim

        self.fc1 = nn.Linear(self.rbf_dim, self.node_dim)
        self.fc2 = nn.Linear(self.node_dim, self.node_dim)
        self.activation = ShiftedSoftPlus()

        return

    def forward(self, x, r):
        r = self.fc1(r)
        r = self.activation(r)
        r = self.fc2(r)
        r = self.activation(r)

        B, N, d = x.size()
        x = x.unsqueeze(1).expand(-1, N, -1, -1)
        x = x * r
        x = torch.sum(x, dim=2)

        return x


class Interaction(nn.Module):
    def __init__(self, rbf_dim, node_dim):
        super(Interaction, self).__init__()
        self.node_dim = node_dim

        self.fc1 = nn.Linear(self.node_dim, self.node_dim)
        self.cfconv = CFConv(rbf_dim, node_dim)
        self.fc2 = nn.Linear(self.node_dim, self.node_dim)
        self.fc3 = nn.Linear(self.node_dim, self.node_dim)
        self.activation = ShiftedSoftPlus()

        return

    def forward(self, h, r):
        x = self.fc1(h)
        x = self.cfconv(x, r)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = h + x
        return x


class SchNet(nn.Module):
    def __init__(self, rbf_dim, node_num, node_dim):
        super(SchNet, self).__init__()
        self.rbf_dim = rbf_dim
        self.node_num = node_num
        self.node_dim = node_dim

        self.interaction_layers = nn.ModuleList(
            [
                Interaction(self.rbf_dim, self.node_dim),
                Interaction(self.rbf_dim, self.node_dim),
                Interaction(self.rbf_dim, self.node_dim)
            ]
        )
        self.layers = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            ShiftedSoftPlus(),
            nn.Linear(self.node_dim, self.node_dim)
        )
        # TODO: need to double check
        self.atom2graph = nn.Linear(self.node_dim, 1)
        return

    def represent(self, h, r):
        x = h
        for interaction_ in self.interaction_layers:
            x = interaction_(x, r)
        x = self.layers(x)
        # TODO: need to double check
        x = torch.sum(x, dim=1)
        return x

    def forward(self, h, r):
        x = self.represent(h, r)
        # TODO: need to double check
        x = self.atom2graph(x)
        return x