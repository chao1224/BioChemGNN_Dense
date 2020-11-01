import os
import numpy as np
from itertools import product


def extract_classification_values(file_path):
    train_ROC, train_PRC, test_ROC, test_PRC = None, None, None, None
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Train'):
                line = line.strip().split('\t')
                if line[1] == 'Mean ROCAUC':
                    train_ROC = float(line[-1])
                elif line[1] == 'Mean PRCAUC':
                    train_PRC = float(line[-1])
            elif line.startswith('Test'):
                line = line.strip().split('\t')
                if line[1] == 'Mean ROCAUC':
                    test_ROC = float(line[-1])
                elif line[1] == 'Mean PRCAUC':
                    test_PRC = float(line[-1])

    return train_ROC, train_PRC, test_ROC, test_PRC


def extract_regression_values(file_path):
    train_RMSE, train_MAE, test_RMSE, test_MAE = None, None, None, None
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Train'):
                line = line.strip().split('\t')
                if line[1] == 'Mean RMSE':
                    train_RMSE = float(line[-1])
                elif line[1] == 'Mean MAE':
                    train_MAE = float(line[-1])
            elif line.startswith('Test'):
                line = line.strip().split('\t')
                if line[1] == 'Mean RMSE':
                    test_RMSE = float(line[-1])
                elif line[1] == 'Mean MAE':
                    test_MAE = float(line[-1])

    return train_RMSE, train_MAE, test_RMSE, test_MAE


def extract_values(task, file_path):
    if task in ['bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21', 'toxcast']:
        return extract_classification_values(file_path)
    else:
        return extract_regression_values(file_path)


def get_first_line(file_path):
    if os.path.exists(file_path):
        f = open(file_path, 'r')
        for line in f:
            return line
    else:
        return None


def get_missing(model, task, mode):
    count = config[model]
    for i in range(count):
        for seed in seed_list:
            try:
                file_path = '{}/{}/{}/{}.out'.format(task, model, seed, i)
                train_RMSE, train_MAE, test_RMSE, test_MAE = extract_values(task, file_path)

                assert train_RMSE is not None
                assert train_MAE is not None
                assert test_RMSE is not None
                assert test_MAE is not None
            except:
                # print('=========\tinvalid {}/{}/{}/{}.out'.format(task, model, seed, i))
                first_line = get_first_line(file_path)
                # print('=========\tfirst line', first_line)

                if first_line is not None:
                    print('\"{} {} {} {}\"'.format(task, model, seed, i))
                    print('=========\tfirst line', first_line)
                else:
                    print('=========\tinvalid {}/{}/{}/{}.out'.format(task, model, seed, i))

    return


def get_top_K_classification(model, task, eval_metric):
    print('{} {}'.format(model, task))
    count = config[model]
    record = []
    for i in range(count):
        values_ROCAUC, values_PRCAUC = [], []
        for seed in seed_list:
            try:
                file_path = '{}/{}/{}/{}.out'.format(task, model, seed, i)
                train_ROCAUC, train_PRCAUC, test_ROCAUC, test_PRCAUC = extract_values(task, file_path)

                assert train_ROCAUC is not None
                assert train_PRCAUC is not None
                assert test_ROCAUC is not None
                assert test_PRCAUC is not None

                values_ROCAUC.append(test_ROCAUC)
                values_PRCAUC.append(test_PRCAUC)
            except:
                print('invalid {} {}'.format(i, seed))

        value_ROCAUC = np.mean(values_ROCAUC)
        value_PRCAUC = np.mean(values_PRCAUC)
        if eval_metric == 'ROCAUC' and not np.isnan(value_ROCAUC):
            record.append([i, value_ROCAUC])
        elif eval_metric == 'PRCAUC' and not np.isnan(value_PRCAUC):
            record.append([i, value_PRCAUC])

    if len(record) == 0:
        print('invalid')
    else:
        record = sorted(record, key=lambda x: x[1], reverse=True)
        record = np.array(record)
        print('top {} index: {}'.format(top_k, record[:top_k, 0]))
        print('top {} {}: {}'.format(top_k, eval_metric, ', '.join(record[:top_k, 1].astype(str))))
    print()
    print()
    print()

    return


def get_top_K_regression(model, task, eval_metric):
    print('{} {}'.format(model, task))
    count = config[model]
    record = []
    for i in range(count):
        values_RMSE, values_MAE = [], []
        for seed in seed_list:
            try:
                file_path = '{}/{}/{}/{}.out'.format(task, model, seed, i)
                train_RMSE, train_MAE, test_RMSE, test_MAE = extract_values(task, file_path)

                assert train_RMSE is not None
                assert train_MAE is not None
                assert test_RMSE is not None
                assert test_MAE is not None

                values_RMSE.append(test_RMSE)
                values_MAE.append(test_MAE)
            except:
                # print('invalid {} {}'.format(i, seed))
                continue

        value_RMSE = np.mean(values_RMSE)
        value_MAE = np.mean(values_MAE)
        if eval_metric == 'RMSE' and not np.isnan(value_RMSE):
            record.append([i, value_RMSE])
        elif eval_metric == 'MAE' and not np.isnan(value_MAE):
            record.append([i, value_MAE])

    if len(record) == 0:
        print('invalid')
    else:
        record = sorted(record, key=lambda x:x[1], reverse=False)
        record = np.array(record)
        print('top {} index: {}'.format(top_k, record[:top_k, 0]))
        print('top {} {}: {}'.format(top_k, eval_metric, ', '.join(record[:top_k, 1].astype(str))))
    print()
    print()
    print()

    return


def get_top_K(model, task, mode):
    if mode == 'classification':
        get_top_K_classification(model, task, eval_metric='ROCAUC')
    elif mode == 'regression':
        get_top_K_regression(model, task, eval_metric='MAE')
    return


if __name__ == '__main__':
    config = {
        'ECFP': 56,
        'NEF': 180,
        'DTNN': 144,
        'GCN': 60,
        'GIN': 60,
        'ENN': 192,
        'SchNet': 4,
        'DMPNN': 4,
    }

    model_list = ['ECFP', 'NEF', 'DTNN', 'ENN', 'GCN', 'GIN', 'SchNet', 'DMPNN']
    task_list = ['delaney', 'freesolv', 'lipophilicity', 'cep', 'qm8', 'qm9']

    model_list = ['ECFP', 'DMPNN', 'SchNet']
    task_list = ['bace', 'bbbp', 'delaney', 'freesolv']
    seed_list = [0, 1, 2, 3, 4]
    top_k = 5

    for model, task in product(model_list, task_list):
        if task in ['bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21', 'toxcast']:
            mode = 'classification'
        else:
            mode = 'regression'
        get_missing(model, task, mode)

        # get_top_K(model, task, mode)
