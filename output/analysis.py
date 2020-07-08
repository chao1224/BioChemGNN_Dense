import numpy as np


def extract_regression_values(file_path):
    '''
    On Test Data
    RMSE: 1.3420450687408447
    MAE: 1.0122097730636597
    '''
    trigger_train, trigger_test = False, False
    argument = ''
    train_RMSE, train_MAE, test_RMSE, test_MAE = None, None, None, None
    with open(file_path, 'r') as f:
        for line in f:
            if '--model' in line:
                argument = line.strip()

            if 'Mean RMSE:' in line:
                line = line.strip().split(':')
                if train_RMSE is None:
                    train_RMSE = float(line[1])
                else:
                    test_RMSE = float(line[1])

            if 'Mean MAE:' in line:
                line = line.strip().split(':')
                if train_MAE is None:
                    train_MAE = float(line[1])
                else:
                    test_MAE = float(line[1])

    return argument, train_RMSE, train_MAE, test_RMSE, test_MAE


def extract(argument):
    argument = argument.split('--')
    model_weight = argument[-1]
    output_file = model_weight.replace('model_weight', 'output').replace('.pt', '.out').replace('output_path=', '')
    return output_file


def get_missing():
    print('argument_list=(')
    argument_list = []
    record_list = []
    for model in model_list:
        for task in task_list:
            if (model, task) in config:

                count = config[(model, task)]
                for i in range(count):
                    values_RMSE, values_MAE = [], []
                    for running_index in range(5):
                        argument = None
                        try:
                            file_path = '{}/{}/{}_{}.out'.format(model, task, i, running_index)
                            argument, train_RMSE, train_MAE, test_RMSE, test_MAE = extract_regression_values(file_path)

                            assert train_RMSE is not None
                            assert train_MAE is not None
                            assert test_RMSE is not None
                            assert test_MAE is not None

                            values_RMSE.append(test_RMSE)
                            values_MAE.append(test_MAE)
                        except:
                            print('\"{}\"'.format(argument))
                            argument_list.append(argument)
                            record_list.append(file_path)
    print(')')

    print('output_file_list=(')
    for argument, record in zip(argument_list, record_list):
        try:
            output_file = extract(argument)
        except:
            print('invalid argument {}'.format(argument))
            print('\"{}\"\t{}'.format(output_file, record))
    print(')')

    print(len(argument_list))

    return


def get_top_K():

    for model in model_list:
        for task in task_list:
            if (model, task) in config:
                print('On task: {}\tmodel: {}'.format(task, model))

                count = config[(model, task)]
                record = []
                for i in range(count):
                    values_RMSE, values_MAE = [], []
                    for running_index in range(5):
                        argument = None
                        try:
                            file_path = '{}/{}/{}_{}.out'.format(model, task, i, running_index)
                            argument, train_RMSE, train_MAE, test_RMSE, test_MAE = extract_regression_values(file_path)

                            assert train_RMSE is not None
                            assert train_MAE is not None
                            assert test_RMSE is not None
                            assert test_MAE is not None

                            values_RMSE.append(test_RMSE)
                            values_MAE.append(test_MAE)
                        except:
                            print('invalid {} {}'.format(i, running_index))

                    value_RMSE = np.mean(values_RMSE)
                    value_MAE = np.mean(values_MAE)
                    record.append([i, argument, value_RMSE, value_MAE])

                record = sorted(record, key=lambda x:x[3], reverse=False)
                record = np.array(record)
                print('top {} index: {}'.format(top_k, record[:top_k, 0]))
                print('top {} argument:\n{}'.format(top_k, '\n'.join(record[:top_k, 1])))
                print('top {} RMSE: {}'.format(top_k, ', '.join(record[:top_k, 2])))
                print('top {} MAE: {}'.format(top_k, ', '.join(record[:top_k, 3])))
                print()
                print()
                print()

    return


if __name__ == '__main__':
    config = {
        ('ECFP', 'delaney'): 56,
        ('ECFP', 'freesolv'): 56,
        ('ECFP', 'lipophilicity'): 56,
        ('ECFP', 'cep'): 56,
        ('ECFP', 'qm8'): 56,
        ('ECFP', 'qm9'): 56,

        ('NEF', 'delaney'): 176,
        ('NEF', 'freesolv'): 176,
        ('NEF', 'lipophilicity'): 176,
        ('NEF', 'cep'): 176,
        ('NEF', 'qm8'): 176,
        ('NEF', 'qm9'): 176,

        ('DTNN', 'delaney'): 144,
        ('DTNN', 'freesolv'): 144,
        ('DTNN', 'lipophilicity'): 144,
        ('DTNN', 'cep'): 144,
        ('DTNN', 'qm8'): 144,
        ('DTNN', 'qm9'): 144,

        # # ('enn-s2s', 'delaney'): 4,
        # # ('enn-s2s', 'qm8'): 4,
        # # ('enn-s2s', 'qm9'): 4,

        ('GIN', 'delaney'): 60,
        ('GIN', 'freesolv'): 60,
        ('GIN', 'lipophilicity'): 60,
        ('GIN', 'cep'): 60,
        ('GIN', 'qm8'): 60,
        ('GIN', 'qm9'): 60,

        ('SchNet', 'delaney'): 4,
        ('SchNet', 'freesolv'): 4,
        ('SchNet', 'lipophilicity'): 4,
        ('SchNet', 'cep'): 4,
        ('SchNet', 'qm8'): 4,
        ('SchNet', 'qm9'): 4,

    }

    model_list = ['ECFP', 'NEF', 'DTNN', 'enn-s2s', 'GIN', 'SchNet']
    task_list = ['delaney', 'freesolv', 'lipophilicity', 'cep', 'qm8', 'qm9']
    top_k = 5

    # get_top_K()
    get_missing()
