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
from dataset import *
from models.ecfp import *
from models.nef import *
from models.schnet import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=str, default='cpu')
parser.add_argument('--gpu', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--k_fold', type=int, default=5)
parser.add_argument('--running_index', type=int, default=0)
parser.add_argument('--representation_analysis', dest='representation_analysis', action='store_true')
parser.add_argument('--no_representation_analysis', dest='representation_analysis', action='store_false')
parser.set_defaults(representation_analysis=False)

parser.add_argument('--fine_tuning', dest='fine_tuning', action='store_true')
parser.add_argument('--no_fine_tuning', dest='fine_tuning', action='store_false')
parser.set_defaults(fine_tuning=False)
parser.add_argument('--pre_trained_model_path', type=str, default='')

parser.add_argument('--task', type=str, default='delaney', choices=[
    'tox21', 'clintox', 'muv', 'hiv', 'pcba',
    'delaney', 'malaria', 'cep', 'qm7', 'qm8', 'qm9'])
parser.add_argument('--model', type=str, default='ECFP', choices=[
    'ECFP', 'NEF', 'DTNN', 'enn-s2s', 'GIN', 'SchNet', 'DimNet'
])
parser.add_argument('--model_weight_dir', type=str, default=None)
parser.add_argument('--model_weight_path', type=str, default=None)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.003)
parser.add_argument('--weight_decay', type=float, default=0.)

# for ECFP
parser.add_argument('--fp_radius', type=int, default=2)
parser.add_argument('--fp_length', type=int, default=1024)
parser.add_argument('--fp_hiddden_dim', type=int, nargs='*', default=[512, 128, 32])

# for NEF
parser.add_argument('--nef_fp_length', type=int, default=1024)
parser.add_argument('--nef_fp_hiddden_dim', type=int, nargs='*', default=[512, 128, 32])


# for SchNet
parser.add_argument('--schnet_low', type=float, default=0.)
parser.add_argument('--schnet_high', type=float, default=30.)
parser.add_argument('--schnet_gap', type=float, default=0.1)


args = parser.parse_args()


config_task2dataset = {
    'delaney': DelaneyDataset,
    'qm9': QM9Dataset,
    'mu': QM9Dataset, 'alpha': QM9Dataset, 'homo': QM9Dataset, 'lumo': QM9Dataset, 'gap': QM9Dataset, 'r2': QM9Dataset,
    'zpve': QM9Dataset, 'cv': QM9Dataset, 'u0': QM9Dataset, 'u298': QM9Dataset, 'h298': QM9Dataset, 'g298': QM9Dataset
}

config_task2tasklist = {
    'delaney': ['delaney'],
    'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298'],
    'mu': ['mu'], 'alpha': ['alpha'], 'homo': ['homo'], 'lumo': ['lumo'], 'gap': ['gap'], 'r2': ['r2'],
    'zpve': ['zpve'], 'cv': ['cv'], 'u0': ['u0'], 'u298': ['u298'], 'h298': ['h298'], 'g298': ['g298'],
}

config_model = {
    'ECFP': ECFPNetwork,
    'NEF': NeuralFingerprint,
    'SchNet': SchNet,
}

config_max_atom_num = {
    'delaney': 56,
    'qm9': 40, 'mu': 40, 'alpha': 40, 'homo': 40, 'lumo': 40, 'gap': 40, 'r2': 40, 'zpve': 40, 'cv': 40, 'u0': 40, 'u298': 40, 'h298': 40, 'g298': 40
}


def get_model_prediction(batch):
    if args.model == 'ECFP':
        ECFP = batch[0].float().to(args.device)
        y_predicted = model(ECFP)
        y_predicted = y_predicted

    elif args.model == 'SchNet':
        node_feature, _, _, distance_list = batch[0], batch[1], batch[2], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)

        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)

        y_predicted = model(node_feature, distance_list)
        y_predicted = y_predicted

    else:
        raise NotImplementedError

    return y_predicted


def train(dataloader, optimizer, criterion, epochs):
    model.train()
    task_num = len(args.task_list)

    for epoch in range(epochs):
        start_time = time.time()
        loss = 0
        for batch in dataloader:
            y_actual = batch[-1].float().to(args.device)
            optimizer.zero_grad()
            y_predicted = get_model_prediction(batch)

            loss_train = 0
            for task_idx in range(task_num):
                loss_train += criterion(y_predicted[:, task_idx], y_actual[:, task_idx])
            loss_train.backward()
            optimizer.step()
            loss += loss_train.item()
        print('Epoch: {:04d}\tLoss Train: {:.5f}\ttime: {:.4f}s'.format(epoch+1, loss/len(dataloader), time.time()-start_time))
    print()
    print()
    return


def test(dataloader, metrics):
    model.eval()
    task_num = len(args.task_list)
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

        metric2value_list = {}
        for metric_name, metric_func in metrics.items():
            metric2value_list[metric_name] = []
            for task_idx in range(task_num):
                value = metric_func(y_predicted[:, task_idx], y_actual[:, task_idx])
                metric2value_list[metric_name].append(value.to(args.cpu).data)

        for metric_name, metric_func in metrics.items():
            value_list = np.array(metric2value_list[metric_name])
            print('All {}: {}'.format(metric_name, ','.join(['{}'.format(x) for x in value_list])))
            print('Mean {}: {}'.format(metric_name, np.mean(value_list)))
        print()
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


def load_model(model, weight_file):
    model.load_state_dict(torch.load(weight_file))
    return


if __name__ == '__main__':
    args.device = torch.device(args.gpu if torch.cuda.is_available() else args.cpu)
    args.task_list = config_task2tasklist[args.task]
    args.task_num = len(args.task_list)
    if args.model_weight_dir is None:
        args.model_weight_dir = 'model_weight/{}'.format(args.task)
        args.model_weight_path = '{}/{}_{}.pt'.format(args.model_weight_dir, args.model, args.running_index)
    print('arguments:\n{}\n'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    kwargs = {'task': args.task, 'task_list': args.task_list, 'model': args.model,
              'max_atom_num': config_max_atom_num[args.task], 'seed': args.seed}
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
        model = config_model[args.model](ECFP_dim=args.fp_length, hidden_dim=args.fp_hiddden_dim, output_dim=args.task_num)
    elif args.model == 'GIN':
        model = config_model[args.model](dataset.node_feature_dim, [256, 256, 256])
        readout = layers.SumReadout()
    elif args.model == 'SchNet':
        rbf_dim = get_RBF_dimension(low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        model = config_model[args.model](
            rbf_dim=rbf_dim, node_num=config_max_atom_num[args.task],
            node_dim=dataset.node_feature_dim, output_dim=args.task_num)
    else:
        raise ValueError

    print('model\n{}\n'.format(model))
    if args.fine_tuning:
        load_model(model, args.pre_trained_model_path)
    model.to(args.device)

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

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train(dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, epochs=args.epochs)

    print('Eva On Training Data')
    test(dataloader=train_dataloader, metrics=metrics)
    print('Eval On Test Data')
    test(dataloader=test_dataloader, metrics=metrics)

    if args.representation_analysis:
        print('Analysis On Training Data')
        analyze(dataloader=train_dataloader, mode='train')
        print('Analysis On Test Data')
        analyze(dataloader=test_dataloader, mode='test')

    save_model(model, dir_=args.model_weight_dir, path_=args.model_weight_path)
