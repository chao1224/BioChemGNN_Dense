import argparse
import time
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
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
from models import *
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

parser.add_argument('--task', type=str, default='chembl', choices=[
    'zinc_2m', 'chembl',
    'bbbp', 'bace',
    'tox21', 'clintox', 'muv', 'hiv', 'pcba',
    'delaney', 'freesolv', 'lipophilicity', 'malaria', 'cep', 'qm7', 'qm7b',
    'qm8', 'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM',
    'qm9', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
])
parser.add_argument('--model', type=str, default='GIN', choices=[
    'ECFP', 'NEF', 'Weave', 'GG-NN', 'DTNN', 'ENN', 'GIN', 'SchNet', 'DimNet'
])
parser.add_argument('--input_model_weight_path', type=str, default=None)
parser.add_argument('--output_model_weight_path', type=str, required=True)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.)

# for ECFP
parser.add_argument('--fp_radius', type=int, default=2)
parser.add_argument('--fp_length', type=int, default=1024)
parser.add_argument('--fp_hiddden_dim', type=int, nargs='*', default=[512, 128, 32])

# for NEF
parser.add_argument('--nef_fp_length', type=int, default=50)
parser.add_argument('--nef_fp_hidden_dim', type=int, nargs='*', default=[20, 20, 20, 20])
parser.add_argument('--nef_fc_hiddden_dim', type=int, nargs='*', default=[100])

# for DTNN
parser.add_argument('--dtnn_low', type=float, default=0.)
parser.add_argument('--dtnn_high', type=float, default=30.)
parser.add_argument('--dtnn_gap', type=float, default=0.1)
parser.add_argument('--dtnn_hidden_dim', type=int, nargs='*', default=[16, 16])
parser.add_argument('--dtnn_fc_hidden_dim', type=int, nargs='*', default=[128])

# for ENN
parser.add_argument('--enn_fc_dim', type=int, nargs='*', default=[128, 256, 128])
parser.add_argument('--enn_hidden_dim', type=int, default=32)
parser.add_argument('--enn_gru_layer_num', type=int, default=1)
parser.add_argument('--enn_layer_num', type=int, default=3)
parser.add_argument('--enn_readout_func', type=str, choices=['set2set', 'sum'], default='set2set')
parser.add_argument('--enn_set2set_processing_steps', type=int, default=3)
parser.add_argument('--enn_set2set_num_layers', type=int, default=1)
parser.add_argument('--enn_use_distance', dest='enn_use_distance', action='store_true')
parser.add_argument('--enn_no_distance', dest='enn_use_distance', action='store_false')
parser.set_defaults(enn_use_distance=True)

# for GIN
parser.add_argument('--gin_hidden_dim', type=int, nargs='*', default=[256, 256])
parser.add_argument('--gin_activation', type=str, default=None)
parser.add_argument('--gin_epsilon', type=float, default=0.)

# for SchNet
parser.add_argument('--schnet_low', type=float, default=0.)
parser.add_argument('--schnet_high', type=float, default=30.)
parser.add_argument('--schnet_gap', type=float, default=0.1)


args = parser.parse_args()


config_task2dataset = {
    'zinc_2m': ZINC2MDataset,
    'chembl': ChEMBLDataset,
    'bbbp': BBBPDataset,
    'bace': BACEDataset,
    'delaney': DelaneyDataset,
    'freesolv': FreeSolvDataset,
    'lipophilicity': LipophilicityDataset,
    'cep': CEPDataset,
    'qm7': QM7Dataset,
    'qm8': QM8Dataset,
    'E1-CC2': QM8Dataset, 'E2-CC2': QM8Dataset, 'f1-CC2': QM8Dataset, 'f2-CC2': QM8Dataset, 'E1-PBE0': QM8Dataset,
    'E2-PBE0': QM8Dataset, 'f1-PBE0': QM8Dataset, 'f2-PBE0': QM8Dataset, 'E1-CAM': QM8Dataset, 'E2-CAM': QM8Dataset,
    'f1-CAM': QM8Dataset, 'f2-CAM': QM8Dataset,
    'qm9': QM9Dataset,
    'mu': QM9Dataset, 'alpha': QM9Dataset, 'homo': QM9Dataset, 'lumo': QM9Dataset, 'gap': QM9Dataset, 'r2': QM9Dataset,
    'zpve': QM9Dataset, 'cv': QM9Dataset, 'u0': QM9Dataset, 'u298': QM9Dataset, 'h298': QM9Dataset, 'g298': QM9Dataset
}

config_task2task_list = {
    'qm8': ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],
    'qm9': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298'],
}

config_model = {
    'ECFP': ECFPNetwork,
    'NEF': NeuralFingerprint,
    'ENN': EdgeNeuralNetwork,
    'GIN': GraphIsomorphismNetwork,
    'DTNN': DeepTensorNeuralNetwork,
    'SchNet': SchNet,
}

config_max_atom_num = {
    'zinc_2m': 10,
    'chembl': 70,
    'bbbp': 132,
    'bace': 97,
    'delaney': 56,
    'freesolv': 24,
    'lipophilicity': 115,
    'cep': 35,
    'qm7': 3,
    'qm7b': 3,
    'qm8': 26, 'E1-CC2': 26, 'E2-CC2': 26, 'f1-CC2': 26, 'f2-CC2': 26, 'E1-PBE0': 26, 'E2-PBE0': 26, 'f1-PBE0': 26, 'f2-PBE0': 26, 'E1-CAM': 26, 'E2-CAM': 26, 'f1-CAM': 26, 'f2-CAM': 26,
    'qm9': 29, 'mu': 29, 'alpha': 29, 'homo': 29, 'lumo': 29, 'gap': 29, 'r2': 29, 'zpve': 29, 'cv': 29, 'u0': 29, 'u298': 29, 'h298': 29, 'g298': 29
}


def get_model_prediction(batch):
    if args.model == 'ECFP':
        ECFP = batch[0].float().to(args.device)
        y_predicted = model(ECFP)
        y_predicted = y_predicted

    elif args.model == 'NEF':
        node_feature, adjacency_matrix = batch[0], batch[2]
        node_feature = node_feature.float().to(args.device)
        adjacency_matrix = adjacency_matrix.float().to(args.device)
        y_predicted = model(node_feature, adjacency_matrix)

    elif args.model == 'ENN':
        node_feature, edge_feature, adjacency_matrix, distance_matrix = batch[0], batch[1], batch[2], batch[3]
        edge_feature = edge_feature.float()
        if args.enn_use_distance:
            distance_matrix = distance_matrix.float().unsqueeze(3)
            edge_feature = torch.cat((edge_feature, distance_matrix), dim=3)
        node_feature = node_feature.float().to(args.device)
        edge_feature = edge_feature.float().to(args.device)
        adjacency_matrix = adjacency_matrix.float().to(args.device)
        y_predicted = model(node_feature, edge_feature, adjacency_matrix)

    elif args.model == 'GIN':
        node_feature, adjacency_matrix = batch[0], batch[2]
        node_feature = node_feature.float().to(args.device)
        adjacency_matrix = adjacency_matrix.float().to(args.device)
        y_predicted = model(node_feature, adjacency_matrix)

    elif args.model == 'DTNN':
        node_feature, distance_list = batch[0], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)
        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        y_predicted = model(node_feature, distance_list)

    elif args.model == 'SchNet':
        node_feature, distance_list = batch[0], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)
        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        y_predicted = model(node_feature, distance_list)

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
                if isinstance(value, np.float64):
                    metric2value_list[metric_name].append(value)
                else:
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

    elif args.model == 'NEF':
        node_feature, adjacency_matrix = batch[0], batch[2]
        node_feature = node_feature.float().to(args.device)
        adjacency_matrix = adjacency_matrix.float().to(args.device)
        y_representation = model.represent(node_feature, adjacency_matrix)
        y_representation = y_representation

    elif args.model == 'ENN':
        node_feature, edge_feature, adjacency_matrix, distance_matrix = batch[0], batch[1], batch[2], batch[3]
        edge_feature = edge_feature.float()
        if args.enn_use_distance:
            distance_matrix = distance_matrix.float().unsqueeze(3)
            edge_feature = torch.cat((edge_feature, distance_matrix), dim=3)
        node_feature = node_feature.float().to(args.device)
        edge_feature = edge_feature.float().to(args.device)
        adjacency_matrix = adjacency_matrix.float().to(args.device)
        y_representation = model.represent(node_feature, edge_feature, adjacency_matrix)

    elif args.model == 'GIN':
        node_feature, adjacency_matrix = batch[0], batch[2]
        node_feature = node_feature.float().to(args.device)
        adjacency_matrix = adjacency_matrix.float().to(args.device)
        y_representation = model.represent(node_feature, adjacency_matrix)
        y_representation = y_representation

    elif args.model == 'DTNN':
        node_feature, distance_list = batch[0], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)
        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        y_representation = model.represent(node_feature, distance_list)

    elif args.model == 'SchNet':
        node_feature, distance_list = batch[0], batch[3]
        node_feature = node_feature.float().to(args.device)
        distance_list = distance_list.float().to(args.device)
        distance_list = RBFExpansion(distance_list.unsqueeze(3), low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        y_representation = model.represent(node_feature, distance_list)

    else:
        raise NotImplementedError

    return y_representation


def test_representation(dataloader):
    model.eval()
    y_represent = []

    with torch.no_grad():
        start_time = time.time()
        for batch in dataloader:
            y_represent_ = get_model_representation(batch)
            print(y_represent_.size())

        y_represent = torch.cat(y_represent)
        print(y_represent.size())
        print('time: {:.4f}s'.format(time.time()-start_time))

    return


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
    if args.task in config_task2task_list:
        args.task_list = config_task2task_list[args.task]
    else:
        args.task_list = [args.task]
    args.task_num = len(args.task_list)
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
    if args.task in [
        'qm8', 'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM',
        'qm9', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
    ]:
        kwargs['node_feature_func'] = 'explicit_property_prediction'
    else:
        kwargs['node_feature_func'] = 'property_prediction'
    kwargs['edge_feature_func'] = 'default'

    if args.task in ['pppb', 'bace']:
        kwargs['k_fold'] = 'StratifiedKFold'
    else:
        kwargs['k_fold'] = 'KFold'

    dataset = config_task2dataset[args.task](**kwargs)

    train_indices, test_indices = split_into_KFold(dataset=dataset, k=args.k_fold, index=args.running_index, **kwargs)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.model == 'ECFP':
        model = config_model[args.model](ECFP_dim=args.fp_length, hidden_dim=args.fp_hiddden_dim, output_dim=args.task_num)
    elif args.model == 'NEF':
        model = config_model[args.model](
            node_feature_dim=dataset.node_feature_dim,
            nef_fp_hidden_dim=args.nef_fp_hidden_dim, nef_fp_length=args.nef_fp_length,
            fc_hidden_dim=args.nef_fc_hiddden_dim, output_dim=args.task_num
        )
    elif args.model == 'DTNN':
        rbf_dim = get_RBF_dimension(low=args.dtnn_low, high=args.dtnn_high, gap=args.dtnn_gap)
        model = config_model[args.model](
            node_feature_dim=dataset.node_feature_dim, rbf_dim=rbf_dim,
            hidden_dim=args.dtnn_hidden_dim, fc_hidden_dim=args.dtnn_fc_hidden_dim,
            output_dim=args.task_num
        )
    elif args.model == 'ENN':
        if args.enn_use_distance:
            edge_feature_dim = dataset.edge_feature_dim + 1
        else:
            edge_feature_dim = dataset.edge_feature_dim
        model = config_model[args.model](
            node_feature_dim=dataset.node_feature_dim, edge_feature_dim=edge_feature_dim,
            hidden_dim=args.enn_hidden_dim, fc_dim=args.enn_fc_dim,
            gru_layer_num=args.enn_gru_layer_num, enn_layer_num=args.enn_layer_num,
            readout_func=args.enn_readout_func,
            set2set_processing_steps=args.enn_set2set_processing_steps, set2set_num_layers=args.enn_set2set_num_layers,
            output_dim=args.task_num
        )
    elif args.model == 'GIN':
        model = config_model[args.model](
            node_feature_dim=dataset.node_feature_dim,
            hidden_dim=args.gin_hidden_dim, output_dim=args.task_num,
            activation=args.gin_activation, epsilon=args.gin_epsilon)
    elif args.model == 'SchNet':
        rbf_dim = get_RBF_dimension(low=args.schnet_low, high=args.schnet_high, gap=args.schnet_gap)
        model = config_model[args.model](
            rbf_dim=rbf_dim, node_num=config_max_atom_num[args.task],
            node_dim=dataset.node_feature_dim, output_dim=args.task_num
        )
    else:
        raise ValueError('Model {} not included.'.format(args.model))

    print('model\n{}\n'.format(model))
    if args.input_model_weight_path is not None:
        load_model(model, args.input_model_weight_path)
    model.to(args.device)

    if args.task in [
        'delaney', 'freesolv', 'lipophilicity', 'cep', 'qm7', 'qm7b',
        'qm8', 'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM',
        'qm9', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
    ]:
        mode = 'regression'
        criterion = nn.MSELoss()
        metrics = {'RMSE': root_mean_squared_error, 'MAE': mean_absolute_error}
    elif args.task in ['bbbp', 'bace', 'hiv']:
        mode = 'classification'
        criterion = nn.BCEWithLogitsLoss()
        metrics = {'ROCAUC': area_under_roc, 'PRCAUC': area_under_prc}
    else:
        raise ValueError('Task {} not included.'.format(args.task))
    kwargs['mode'] = mode

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # train(dataloader=train_dataloader, optimizer=optimizer, criterion=criterion, epochs=args.epochs)
    #
    # print('Eval On Training Data')
    # test(dataloader=train_dataloader, metrics=metrics)
    # print('Eval On Test Data')
    # test(dataloader=test_dataloader, metrics=metrics)
    #
    # if args.representation_analysis:
    #     print('Analysis On Training Data')
    #     analyze(dataloader=train_dataloader, mode='train')
    #     print('Analysis On Test Data')
    #     analyze(dataloader=test_dataloader, mode='test')

    test_representation(dataloader=train_dataloader)

    save_model(model, path_=args.output_model_weight_path)
