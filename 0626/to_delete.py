import torch
from torch import nn


x = torch.tensor([
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11]
])
N = 3

a = x.unsqueeze(2).expand(-1, -1, N).reshape(-1, 4)
b = x.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)

print(x.size())
print('x:\n{}'.format(x))#.reshape(-1, D)

print('a:\n{}'.format(a))#.reshape(-1, D)
print(a[0, 0])

print('b:\n{}'.format(b))#.reshape(-1, D)
print(b[0, 0])
