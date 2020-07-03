import argparse
import time
import os
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from matplotlib.colors import LinearSegmentedColormap
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data import SubsetRandomSampler

from data import *
from models.schnet import *
from models.ecfp import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=str, default='cpu')
parser.add_argument('--gpu', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--k_fold', type=int, default=5)
parser.add_argument('--running_index', type=int, default=0)

parser.add_argument('--task', type=str, default='KG', choices=[
    'KG'])
parser.add_argument('--model', type=str, default='ECFP', choices=[
    'ECFP', 'GCNN', 'MPNN', 'GIN', 'SchNet', 'DimNet'
])
parser.add_argument('--model_weight_dir', type=str, default='pre_model_weight/{}')
parser.add_argument('--model_weight_path', type=str, default='{}/{}.pt')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--weight_decay', type=float, default=0.)

# for ECFP
parser.add_argument('--fp_radius', type=int, default=2)
parser.add_argument('--fp_length', type=int, default=1024)
parser.add_argument('--fp_hiddden_dim', type=int, nargs='*', default=[512, 128, 32])

# for schnet
parser.add_argument('--schnet_low', type=float, default=0.)
parser.add_argument('--schnet_high', type=float, default=30.)
parser.add_argument('--schnet_gap', type=float, default=0.1)


args = parser.parse_args()


config_task2dataset = {
    'KG': StitchDDIDataset,
}

config_model = {
    'ECFP': ECFPNetwork,
    'SchNet': SchNet,
}

config_max_atom_num = {
    'KG': 10
}



def train(dataloader, optimizer, criterion, epochs):
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        loss = 0
        for batch in dataloader:
            drug = batch[0].float().to(args.device)
            pos = batch[1].float().to(args.device)
            neg = batch[2].float().to(args.device)
            B = drug.size()[0]
            optimizer.zero_grad()

            drug = model.represent(drug)
            pos = model.represent(pos)
            neg = model.represent(neg)

            pos_pair = F.cosine_similarity(drug, pos)
            neg_pair = F.cosine_similarity(drug, neg)
            correlation = torch.stack((pos_pair, neg_pair), dim=1)
            label = torch.ones(B)
            label = label.long().to(args.device)

            loss_train = criterion(correlation, label)

            loss_train.backward()
            optimizer.step()
            loss += loss_train.item()
        print('Epoch: {:04d}\tLoss Train: {:.5f}\ttime: {:.4f}s'.format(epoch+1, loss/len(dataloader), time.time()-start_time))
    print()
    return


def save_model(model, dir_, path_):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    torch.save(model.state_dict(), path_)
    return


if __name__ == '__main__':
    args.device = torch.device(args.gpu if torch.cuda.is_available() else args.cpu)
    args.model_weight_dir = args.model_weight_dir.format(args.task)
    args.model_weight_path = args.model_weight_path.format(args.model_weight_dir, args.model)
    print('arguments:\n{}\n'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    kwargs = {'task': args.task, 'model': args.model, 'max_atom_num': config_max_atom_num[args.task], 'seed': args.seed}
    if args.model == 'ECFP':
        kwargs['fp_radius'] = args.fp_radius
        kwargs['fp_length'] = args.fp_length
    if args.task in ['qm8', 'qm9']:
        kwargs['node_feature_func'] = 'explicit_property_prediction'
    else:
        kwargs['node_feature_func'] = 'property_prediction'
    kwargs['edge_feature_func'] = 'default'

    if args.task in ['hiv']:
        kwargs['k_fold'] = 'StratifiedKFold'
    else:
        kwargs['k_fold'] = 'KFold'

    dataset = config_task2dataset[args.task](
        **kwargs
    )

    train_indices, test_indices = split_into_KFold(dataset=dataset, k=args.k_fold, index=args.running_index, **kwargs)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.model == 'ECFP':
        model = config_model[args.model](ECFP_dim=args.fp_length, hidden_dim=args.fp_hiddden_dim, output_dim=1)
    elif args.model == 'GIN':
        model = config_model[args.model](dataset.node_feature_dim, [256, 256, 256])
        readout = layers.SumReadout()
    elif args.model == 'SchNet':
        rbf_dim = get_RBF_dimension(low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        model = config_model[args.model](rbf_dim=rbf_dim, node_num=config_max_atom_num[args.task], node_dim=dataset.node_feature_dim)
    else:
        raise ValueError

    print('model\n{}\n'.format(model))

    criterion = nn.CrossEntropyLoss()
    metrics = {'ROCAUC': area_under_roc, 'PRCAUC': area_under_prc}
    kwargs['mode'] = 'classification'
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train(dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, epochs=args.epochs)

    # print('On Training Data')
    # test(dataloader=train_dataloader, metrics=metrics)
    # print('On Test Data')
    # test(dataloader=test_dataloader, metrics=metrics)

    save_model(model, dir_=args.model_weight_dir, path_=args.model_weight_path)
