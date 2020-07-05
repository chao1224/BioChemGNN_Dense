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
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, self.num_layers, batch_first=True)
        return

    def forward(self, embedding):
        batch_size, n = embedding.size()[0], embedding.size()[1]
        if torch.cuda.is_available():
            h = (torch.zeros(self.num_layers, batch_size, self.input_dim).cuda(),
                 torch.zeros(self.num_layers, batch_size, self.input_dim).cuda())
            q_star = torch.zeros(batch_size, 1, self.output_dim).cuda()
        else:
            h = (torch.zeros(self.num_layers, batch_size, self.input_dim),
                 torch.zeros(self.num_layers, batch_size, self.input_dim))
            q_star = torch.zeros(batch_size, 1, self.output_dim)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star, h)
            e = torch.bmm(embedding, torch.transpose(q, 1, 2))
            a = nn.Softmax(dim=1)(e)
            r = torch.sum(a * embedding, dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=2)

        q_star = torch.squeeze(q_star, dim=1)
        return q_star
