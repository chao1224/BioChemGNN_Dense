import numpy as np


def filter_NEF_delaney():
    hyper_set = set()
    count = 0

    nef_fp_length_list = []
    nef_fp_hidden_dim_list = []
    nef_fc_hidden_dim_list = []
    with open('for_filtering_NEF_delaney.out', 'r') as f:
        for line in f:
            if '--model=NEF' in line:
                line = line.strip()
                line = line.split('--')
                nef_fp_length = line[5].replace('nef_fp_length=', '').strip()
                nef_fp_hidden_dim = line[6].replace('nef_fp_hidden_dim', '').strip()
                nef_fc_hidden_dim = line[7].replace('nef_fc_hiddden_dim', '').strip()

                count += 1
                if (nef_fp_length, nef_fp_hidden_dim, nef_fc_hidden_dim) in hyper_set:
                    continue
                hyper_set.add((nef_fp_length, nef_fp_hidden_dim, nef_fc_hidden_dim))

                nef_fp_length_list.append(nef_fp_length)
                nef_fp_hidden_dim_list.append(nef_fp_hidden_dim)
                nef_fc_hidden_dim_list.append(nef_fc_hidden_dim)
                if count >= 50:
                    break

    print('{} hypers in all'.format(len(hyper_set)))
    print('nef_fp_length_list=({})'.format(' '.join(nef_fp_length_list)))
    print('nef_fp_hidden_dim_list=({})'.format(' '.join(
        ['"{}"'.format(x) for x in nef_fp_hidden_dim_list]
    )))
    print('nef_fc_hidden_dim_list=({})'.format(' '.join(
        ['"{}"'.format(x) for x in nef_fc_hidden_dim_list]
    )))

    print(len(nef_fp_length_list), '\t', len(nef_fp_hidden_dim_list), '\t', len(nef_fc_hidden_dim_list))

    return


if __name__ == '__main__':
    filter_NEF_delaney()
