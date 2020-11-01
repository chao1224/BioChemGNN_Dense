import os
from itertools import product


if __name__ == '__main__':
    task_list = [
        'bace', 'bbbp', 'clintox', 'hiv', 'muv', 'sider', 'tox21', 'toxcast',
        'delaney', 'freesolv', 'lipophilicity', 'malaria', 'cep', 'qm8', 'qm9'
    ]

    model_list = ['ECFP', 'NEF', 'DTNN', 'ENN', 'GCN', 'GIN', 'DMPNN', 'SchNet']

    for task in task_list:
        for model in model_list:
            os.makedirs('{}/{}'.format(task, model), exist_ok=True)
            if model == 'ECFP':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]
                fp_hidden_dim_list = [
                    '128 8', '512', '512 128', '512 128 32', '256', '256 64', '256 64 16', '128', '128 16',
                    '64', '64 8', '32', '32 4', '16'
                ]

                index = 0
                for epochs, learning_rate, fp_hidden_dim in product(epochs_list, learning_rate_list, fp_hidden_dim_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={} --fp_hidden_dim {}'.format(task, model, epochs, learning_rate, fp_hidden_dim), file=f)
                    index += 1

            elif model == 'NEF':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]
                nef_fp_length_list = [
                    16, 16, 128, 16, 50, 16, 16, 50, 16, 16, 16, 16, 16, 16, 16, 50, 50, 16,
                    50, 16, 16, 16, 16, 16, 16, 16, 16, 16, 50, 16, 16, 16, 128, 16, 128, 16,
                    16, 50, 50, 50, 50, 16, 16, 50, 50
                ]
                nef_fp_hidden_dim_list = [
                    '512 128 64', '128 128 128 128', '20 20 20 20', '512 128 32', '512 512 512 512',
                    '512 128 64', '128 128 128 128', '128 128 128 128', '128 128 128 128', '512 128 32',
                    '20 20 20 20', '512 128 64', '512 128 64', '512 512 512 512', '512 128 64',
                    '512 512 512 512', '512 128 64', '512 128 32', '512 128 64', '512 128 64', '20 20 20 20',
                    '512 128 64', '512 128 32', '512 128 32', '128 128 128 128', '128 128 128 128',
                    '20 20 20 20', '128 128 128 128', '512 128 32', '512 128 32', '128 128 128 128',
                    '20 20 20 20', '512 128 32', '512 128 32', '512 128 64', '512 128 64', '512 512 512 512',
                    '512 128 64', '512 128 32', '512 128 32', '512 128 64', '512 512 512 512', '512 128 64',
                    '512 128 64', '128 128 128 128'
                ]
                nef_fc_hidden_dim_list = [
                    '512 128 32', '32 4', '32 4', '512 128', '128 8', '256 64 16', '512 128 32', '64 8', '64 8',
                    '512 128 32', '512 128 32', '16', '128 16', '512 128 32', '32 4', '128 16', '32', '64', '32 4',
                    '256 64', '16', '32', '128 8', '256 64', '256 64', '128 16', '32 4', '256 64 16', '64 8',
                    '128 16', '32', '128 8', '32 4', '256 64 16', '32 4', '128', '32 4', '64 8', '64', '256 64 16',
                    '64', '256 64 16', '64 8', '512', '256 64 16'
                ]

                index = 0
                for epochs, learning_rate, (nef_fp_length, nef_fp_hidden_dim, nef_fc_hidden_dim) in product(
                        epochs_list, learning_rate_list, zip(nef_fp_length_list, nef_fp_hidden_dim_list, nef_fc_hidden_dim_list)):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={} --nef_fp_length={} --nef_fp_hidden_dim {} --nef_fc_hidden_dim {}'.format(
                        task, model, epochs, learning_rate, nef_fp_length, nef_fp_hidden_dim, nef_fc_hidden_dim), file=f)
                    index += 1

            elif model == 'DTNN':
                epochs_list = [50, 500]
                learning_rate_list = [0.001, 0.003]
                dtnn_hidden_dim_list = ['64 64 64', '64 32', '32 32 32', '32 16', '16 16', '16']
                dtnn_fc_hidden_dim_list = ['128', '128 8', '64', '16', '8', '']

                index = 0
                for epochs, learning_rate, dtnn_hidden_dim, dtnn_fc_hidden_dim in product(epochs_list, learning_rate_list, dtnn_hidden_dim_list, dtnn_fc_hidden_dim_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={} --dtnn_hidden_dim {} --dtnn_fc_hidden_dim {}'.format(
                        task, model, epochs, learning_rate, dtnn_hidden_dim, dtnn_fc_hidden_dim), file=f)
                    index += 1

            elif model == 'ENN':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]

                enn_hidden_dim_list = [32, 64, 128]
                enn_layer_num_list = [1, 2, 3, 5]
                enn_fc_dim_list = ['256', '128 256 128', '128 256', '128']
                enn_readout_func_list = ['set2set']

                index = 0
                for epochs, learning_rate, enn_hidden_dim, enn_layer_num, enn_fc_dim, enn_readout_func in product(
                        epochs_list, learning_rate_list, enn_hidden_dim_list, enn_layer_num_list, enn_fc_dim_list, enn_readout_func_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={} --enn_hidden_dim={} --enn_layer_num={} --enn_fc_dim {} --enn_readout_func={}'.format(
                        task, model, epochs, learning_rate, enn_hidden_dim, enn_layer_num, enn_fc_dim, enn_readout_func),file=f)
                    index += 1

            elif model == 'GCN':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]
                gcn_hidden_dim_list = [
                    '128 8', '512', '512 128', '512 128 32', '256', '256 64', '256 64 16', '128', '128 16',
                    '64', '64 8', '32', '32 4', '16', '128 128 128'
                ]

                index = 0
                for epochs, learning_rate, gcn_hidden_dim in product(epochs_list, learning_rate_list, gcn_hidden_dim_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={} --gcn_hidden_dim {}'.format(task, model, epochs, learning_rate, gcn_hidden_dim), file=f)
                    index += 1

            elif model == 'GIN':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]
                gin_hidden_dim_list = [
                    '128 8', '512', '512 128', '512 128 32', '256', '256 64', '256 64 16', '128', '128 16',
                    '64', '64 8', '32', '32 4', '16', '128 128 128'
                ]
                gin_epsilon_list=[0]

                index = 0
                for epochs, learning_rate, gin_hidden_dim, gin_epsilon in product(epochs_list, learning_rate_list, gin_hidden_dim_list, gin_epsilon_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={} --gin_hidden_dim {} --gin_epsilon={}'.format(
                        task, model, epochs, learning_rate, gin_hidden_dim, gin_epsilon), file=f)
                    index += 1

            elif model == 'DMPNN':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]

                index = 0
                for epochs, learning_rate in product(epochs_list, learning_rate_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={}'.format(task, model, epochs, learning_rate), file=f)
                    index += 1

            elif model == 'SchNet':
                epochs_list = [100, 1000]
                learning_rate_list = [0.001, 0.003]

                index = 0
                for epochs, learning_rate in product(epochs_list, learning_rate_list):
                    f = open('{}/{}/{}.hyper'.format(task, model, index), 'w')
                    print('--task={} --model={} --epochs={} --learning_rate={}'.format(task, model, epochs, learning_rate), file=f)
                    index += 1

