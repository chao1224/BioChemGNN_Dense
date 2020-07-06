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
            if 'Eval On Training Data':
                trigger_train = True
            if 'Eval On Test Data':
                trigger_test = True
            if '--model' in line:
                argument = line.strip()

            if trigger_test and 'Mean RMSE:' in line:
                line = line.strip().split(':')
                test_RMSE = float(line[1])
            elif trigger_test and 'Mean MAE:' in line:
                line = line.strip().split(':')
                test_MAE = float(line[1])
            elif trigger_train and 'Mean RMSE:' in line:
                line = line.strip().split(':')
                train_RMSE = float(line[1])
            elif trigger_train and 'Mean MAE:' in line:
                line = line.strip().split(':')
                train_MAE = float(line[1])

    return argument, train_RMSE, train_MAE, test_RMSE, test_MAE


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

        # ('enn-s2s', 'delaney'): 4,
        # ('enn-s2s', 'qm8'): 4,
        # ('enn-s2s', 'qm9'): 4,

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

    for model in model_list:
        for task in task_list:
            if (model, task) in config:
                print('On task: {}\tmodel: {}'.format(task, model))

                count = config[(model, task)]
                record = []
                for i in range(count):
                    values_RMSE, values_MAE = [], []
                    try:
                        for running_index in range(5):
                            file_path = '{}/{}/{}_{}.out'.format(model, task, i, running_index)
                            argument, train_RMSE, train_MAE, test_RMSE, test_MAE = extract_regression_values(file_path)
                            values_RMSE.append(test_RMSE)
                            values_MAE.append(test_MAE)
                        value_RMSE = np.mean(values_RMSE)
                        value_MAE = np.mean(values_MAE)

                        record.append([i, argument, value_RMSE, value_MAE])
                    except:
                        print('invalid {}'.format(i))
                record = sorted(record, key=lambda x:x[3], reverse=False)
                record = np.array(record)
                print('top {} index: {}'.format(top_k, record[:top_k, 0]))
                print('top {} argument:\n{}'.format(top_k, '\n'.join(record[:top_k, 1])))
                print('top {} RMSE: {}'.format(top_k, ', '.join(record[:top_k, 2])))
                print('top {} MAE: {}'.format(top_k, ', '.join(record[:top_k, 3])))
                print()
                print()
                print()
