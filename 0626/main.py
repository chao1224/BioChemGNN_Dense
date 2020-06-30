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
from schnet import *
from ecfp import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=str, default='cpu')
parser.add_argument('--gpu', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--k_fold', type=int, default=5)
parser.add_argument('--running_index', type=int, default=0)

parser.add_argument('--task', type=str, default='delaney', choices=[
    'tox21', 'clintox', 'muv', 'hiv', 'pcba',
    'delaney', 'malaria', 'cep', 'qm7', 'qm8', 'qm9'])
parser.add_argument('--model', type=str, default='ECFP', choices=[
    'ECFP', 'GCNN', 'MPNN', 'GIN', 'SchNet', 'DimNet'
])
parser.add_argument('--model_weight_dir', type=str, default='model_weight/{}')
parser.add_argument('--model_weight_path', type=str, default='{}/{}.pt')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--weight_decay', type=float, default=0.)

# for schnet
parser.add_argument('--schnet_low', type=float, default=0.)
parser.add_argument('--schnet_high', type=float, default=30.)
parser.add_argument('--schnet_gap', type=float, default=0.1)


args = parser.parse_args()


config_task2dataset = {
    'delaney': DelaneyDataset,
    'qm9': QM9Dataset,
}

config_model = {
    'ECFP': ECFPNetwork,
    'SchNet': SchNet,
}

config_max_atom_num = {
    'delaney': 56,
    'qm9': 40,
}


def get_model_prediction(batch):
    if args.model == 'ECFP':
        ECFP = batch[0].float().to(args.device)
        y_predicted = model(ECFP)
        y_predicted = y_predicted.squeeze()

    elif args.model == 'SchNet':
        node_feature, _, _, distance_list = batch[0], batch[1], batch[2], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)

        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)

        y_predicted = model(node_feature, distance_list)
        y_predicted = y_predicted.squeeze()

    else:
        raise NotImplementedError

    return y_predicted


def train(dataloader, optimizer, criterion, epochs):
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        loss = 0
        for batch in dataloader:
            y_actual = batch[-1].float().to(args.device)
            optimizer.zero_grad()
            y_predicted = get_model_prediction(batch)
            loss_train = criterion(y_predicted, y_actual)

            loss_train.backward()
            optimizer.step()
            loss += loss_train.item()
        print('Epoch: {:04d}\tLoss Train: {:.5f}\ttime: {:.4f}s'.format(epoch+1, loss/len(dataloader), time.time()-start_time))
    print()
    return


def test(dataloader, metrics):
    model.eval()
    y_actual, y_predicted = [], []

    with torch.no_grad():
        for batch in dataloader:
            y_actual_ = batch[-1].float().to(args.device)
            y_predicted_ = get_model_prediction(batch)
            if mode == 'classification':
                y_predicted_ = torch.sigmoid(y_predicted_)
            y_actual.append(y_actual_)
            y_predicted.append(y_predicted_)

        y_actual = torch.cat(y_actual)
        y_predicted = torch.cat(y_predicted)

        for metric_name, metric_func in metrics.items():
            value = metric_func(y_predicted, y_actual)
            print('{}: {}'.format(metric_name, value))
        print()
    return


def get_model_representation(batch):
    if args.model == 'ECFP':
        ECFP = batch[0].float().to(args.device)
        y_representation = model.represent(ECFP)

    elif args.model == 'SchNet':
        node_feature, _, _, distance_list = batch[0], batch[1], batch[2], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)

        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)

        y_representation = model.represent(node_feature, distance_list)

    else:
        raise NotImplementedError

    return y_representation


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def uniform_distribution(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().log()


def analyze(dataloader, mode):
    model.eval()
    y_represent, y_predicted, y_actual = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            y_actual_ = batch[-1].float().to(args.device)
            y_predicted_ = get_model_prediction(batch)
            if mode == 'classification':
                y_predicted_ = torch.sigmoid(y_predicted_)
            y_represent_ = get_model_representation(batch)

            y_actual.append(y_actual_)
            y_predicted.append(y_predicted_)
            y_represent.append(y_represent_)

        y_actual = torch.cat(y_actual)
        y_predicted = torch.cat(y_predicted)
        y_represent = torch.cat(y_represent)

        values = y_represent.to(args.cpu).data.numpy()
        y_embedded = TSNE(n_components=2, random_state=args.seed).fit_transform(values)

        dir_ = 'figures/{}'.format(args.task)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

        fig, ax = plt.subplots()
        targets = y_actual.to(args.cpu).data.numpy()
        cax = ax.scatter(y_embedded[:, 0], y_embedded[:, 1], s=10, c=targets, alpha=0.9, cmap='YlOrBr')
        fig.colorbar(cax)
        plt.savefig('figures/{}/{}_{}_actual'.format(args.task, args.model, mode), bbox_inches='tight')
        plt.clf()

        fig, ax = plt.subplots()
        targets = y_predicted.to(args.cpu).data.numpy()
        cax = ax.scatter(y_embedded[:, 0], y_embedded[:, 1], s=10, c=targets, alpha=0.9, cmap='YlOrBr')
        fig.colorbar(cax)
        plt.savefig('figures/{}/{}_{}_prediction'.format(args.task, args.model, mode), bbox_inches='tight')
        plt.clf()
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
        kwargs['fp_radius'] = 2
        kwargs['fp_length'] = 1024
    if args.task in ['qm8', 'qm9']:
        kwargs['node_feature_func'] = 'explicit_property_prediction'
    else:
        kwargs['node_feature_func'] = 'property_prediction'
    kwargs['edge_feature_func'] = 'default'

    if args.task in ['hiv']:
        kwargs['k_fold'] = 'StratifiedKFold'
    else:
        kwargs['k_fold'] = 'KFold'

    # TBA: cross validation
    dataset = config_task2dataset[args.task](
        **kwargs
    )

    train_indices, test_indices = split_into_KFold(dataset=dataset, k=args.k_fold, index=args.running_index, **kwargs)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.model == 'ECFP':
        model = config_model[args.model](ECFP_dim=1024, hidden_dim=[128, 8], output_dim=1)
    elif args.model == 'GIN':
        model = config_model[args.model](dataset.node_feature_dim, [256, 256, 256])
        readout = layers.SumReadout()
    elif args.model == 'SchNet':
        rbf_dim = get_RBF_dimension(low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        model = config_model[args.model](rbf_dim=rbf_dim, node_num=config_max_atom_num[args.task], node_dim=dataset.node_feature_dim)
    else:
        raise ValueError

    print('model\n{}\n'.format(model))

    if args.task in ['delaney', 'qm7', 'qm8', 'qm9']:
        mode = 'regression'
        criterion = nn.MSELoss()
        metrics = {'RMSE': root_mean_squared_error, 'MAE': mean_absolute_error}
    elif args.task in ['hiv']:
        mode = 'classification'
        criterion = nn.BCEWithLogitsLoss()
        metrics = {'ROCAUC': area_under_roc, 'PRCAUC': area_under_prc}
    else:
        raise ValueError
    kwargs['mode'] = mode
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train(dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, epochs=args.epochs)

    print('On Training Data')
    test(dataloader=train_dataloader, metrics=metrics)
    print('On Test Data')
    test(dataloader=test_dataloader, metrics=metrics)

    print('On Training Data')
    analyze(dataloader=train_dataloader, mode='train')
    print('On Test Data')
    analyze(dataloader=test_dataloader, mode='test')

    save_model(model, dir_=args.model_weight_dir, path_=args.model_weight_path)
