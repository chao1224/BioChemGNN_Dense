import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import optim

from data import *
from schnet import *
from ecfp import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=str, default='cpu')
parser.add_argument('--gpu', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1337)

parser.add_argument('--task', type=str, default='delaney', choices=[
    'tox21', 'clintox', 'muv', 'hiv', 'pcba',
    'delaney', 'malaria', 'cep', 'qm7', 'qm8', 'qm9'])
parser.add_argument('--model', type=str, default='ECFP', choices=[
    'ECFP', 'GCNN', 'MPNN', 'GIN', 'SchNet', 'DimNet'
])

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


def feed_into_network(batch):
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
            y_predicted = feed_into_network(batch)
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
            y_predicted_ = feed_into_network(batch)
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


def analyze(dataloader):

    return


if __name__ == '__main__':
    args.device = torch.device(args.gpu if torch.cuda.is_available() else args.cpu)
    print('arguments:\n{}\n'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    kwargs = {'task': args.task, 'model': args.model, 'max_atom_num':config_max_atom_num[args.task]}
    if args.model == 'ECFP':
        kwargs['fp_radius'] = 2
        kwargs['fp_length'] = 1024
    if args.task in ['qm8', 'qm9']:
        kwargs['node_feature_func'] = 'explicit_property_prediction'
    else:
        kwargs['node_feature_func'] = 'property_prediction'
    kwargs['edge_feature_func'] = 'default'

    # TBA: cross validation
    dataset = config_task2dataset[args.task](
        **kwargs
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'ECFP':
        model = config_model[args.model](ECFP_dim=1024, hidden_dim=[256, 16], output_dim=1)
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
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train(dataloader=dataloader, optimizer=optimizer, criterion=criterion, epochs=args.epochs)

    print('On Training Data')
    test(dataloader=dataloader, metrics=metrics)
    print('On Test Data')
    test(dataloader=dataloader, metrics=metrics)

    print('On Training Data')
    analyze(dataloader=dataloader)
    print('On Test Data')
    analyze(dataloader=dataloader)

    # print(uniform_loss(torch.FloatTensor([0, 1, 2, 3])), 2)