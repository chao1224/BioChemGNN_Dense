import numpy as np


def extract_regression_values(file_path):
    '''
    On Test Data
    RMSE: 1.3420450687408447
    MAE: 1.0122097730636597
    '''
    trigger_train, trigger_test = False, False
    argument = ''
    train_RMSE, train_MAE, test_RMSE, test_MAE = 0, 0, 0, 0
    with open(file_path, 'r') as f:
        for line in f:
            if 'On Train Data':
                trigger_train = True
            if 'On Test Data':
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
        ('ECFP', 'qm8'): 56,
        ('ECFP', 'qm9'): 56,

        # ('NEF', 'delaney'): 176,
        # ('NEF', 'qm8'): 176,
        # ('NEF', 'qm9'): 176,

        # ('DTNN', 'delaney'): 4,
        # ('DTNN', 'qm8'): 4,
        # ('DTNN', 'qm9'): 4,

        # ('enn-s2s', 'delaney'): 4,
        # ('enn-s2s', 'qm8'): 4,
        # ('enn-s2s', 'qm9'): 4,

        ('GIN', 'delaney'): 60,
        ('GIN', 'qm8'): 60,
        ('GIN', 'qm9'): 60,

        ('SchNet', 'delaney'): 4,
        ('SchNet', 'qm8'): 4,
        ('SchNet', 'qm9'): 4,

    }

    model_list = ['ECFP', 'NEF', 'DTNN', 'enn-s2s', 'GIN', 'SchNet']
    task_list = ['delaney', 'qm8', 'qm9']
    top_k = 5

    for model in model_list:
        for task in task_list:
            if (model, task) in config:
                count = config[(model, task)]
                record = []
                for i in range(count):
                    values_RMSE, values_MAE = [], []
                    for running_index in range(5):
                        file_path = '{}/{}/{}_{}.out'.format(model, task, i, running_index)
                        argument, train_RMSE, train_MAE, test_RMSE, test_MAE = extract_regression_values(file_path)
                        values_RMSE.append(test_RMSE)
                        values_MAE.append(test_MAE)
                    value_RMSE = np.mean(values_RMSE)
                    value_MAE = np.mean(values_MAE)

                    record.append([i, argument, value_RMSE, value_MAE])

                print('On task: {}\tmodel: {}'.format(task, model))
                record = sorted(record, key=lambda x:x[3], reverse=False)
                record = np.array(record)
                print('top {} index: {}'.format(top_k, record[:top_k, 0]))
                print('top {} argument:\n{}'.format(top_k, '\n'.join(record[:top_k, 1])))
                print('top {} RMSE: {}'.format(top_k, ', '.join(record[:top_k, 2])))
                print('top {} MAE: {}'.format(top_k, ', '.join(record[:top_k, 3])))
                print()
                print()
                print()
