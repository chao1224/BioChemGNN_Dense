import torch
from data import *
from utils import *
from utils import _get_explicit_property_prediction_node_feature, _get_property_prediction_node_feature, \
    _get_default_edge_feature, _get_node_dim, _get_edge_dim, _get_atom_distance, _get_atom_distance, \
    _get_max_atom_num_from_smiles_list, _get_max_atom_num_from_molecule_list


class QM9Dataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super(QM9Dataset, self).__init__()
        self.model = kwargs['model']
        task_list = kwargs['task_list']

        if self.model == 'ECFP':
            csv_file = './datasets/qm9.csv'
            smiles_list, self.task_label_list = from_2Dcsv(csv_file, smiles_field='smiles', task_list_field=task_list)
            print('max atom number: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
            kwargs['representation'] = 'smiles'
            self.data = transform(smiles_list, **kwargs)
        else:
            sdf_file = './datasets/qm9.sdf'
            csv_file = './datasets/qm9.sdf.csv'
            molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
            print('max atom number: {}'.format(_get_max_atom_num_from_molecule_list(molecule_list)))
            _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=task_list)
            kwargs['representation'] = 'molecule'
            self.data = transform(molecule_list, **kwargs)

        return

    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            target = self.task_label_list[index]
            return ecfp, target
        else: # is graph
            node_feature, edge_feature, adjacency_list, distance_list = self.data[index]
            target = self.task_label_list[index]
            return node_feature, edge_feature, adjacency_list, distance_list, target

    @property
    def node_feature_dim(self):
        return self.data[0][0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[1][0].shape[-1]

    def __len__(self):
        return len(self.data)


class QM8Dataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super(QM8Dataset, self).__init__()
        self.model = kwargs['model']
        task_list = kwargs['task_list']

        if self.model == 'ECFP':
            csv_file = './datasets/qm8.csv'
            smiles_list, self.task_label_list = from_2Dcsv(csv_file, smiles_field='smiles', task_list_field=task_list)
            kwargs['representation'] = 'smiles'
            self.data = transform(smiles_list, **kwargs)
            # print('max atom number: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        else:
            sdf_file = './datasets/qm8.sdf'
            csv_file = './datasets/qm8.sdf.csv'
            molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
            _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=task_list)
            kwargs['representation'] = 'molecule'
            self.data = transform(molecule_list, **kwargs)
            # print('max atom number: {}'.format(_get_max_atom_num_from_molecule_list(molecule_list)))

        return

    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            target = self.task_label_list[index]
            return ecfp, target
        else: # is graph
            node_feature, edge_feature, adjacency_list, distance_list = self.data[index]
            target = self.task_label_list[index]
            return node_feature, edge_feature, adjacency_list, distance_list, target

    @property
    def node_feature_dim(self):
        return self.data[0][0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[1][0].shape[-1]

    def __len__(self):
        return len(self.data)


class DelaneyDataset(torch.utils.data.Dataset):
    given_target = 'measured log solubility in mols per litre'
    task = 'delaney'

    def __init__(self, **kwargs):
        super(DelaneyDataset, self).__init__()
        self.model = kwargs['model']

        file_name = './datasets/delaney-processed.csv'
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return

    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            target = self.task_label_list[index]
            return ecfp, target
        else: # is graph
            node_feature, edge_feature, adjacency_list, distance_list = self.data[index]
            target = self.task_label_list[index]
            return node_feature, edge_feature, adjacency_list, distance_list, target

    @property
    def node_feature_dim(self):
        return self.data[0][0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[1][0].shape[-1]

    def __len__(self):
        return len(self.data)


class StitchDDIDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(StitchDDIDataset, self).__init__()
        self.model = kwargs['model']

        drug2pos, drug2neg = defaultdict(list), defaultdict(list)
        DDI_file = '../datasets/STITCH_DDI/DDI.tsv'
        drug_set = set()
        with open(DDI_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                drug, pos, neg = line[0], line[1], line[2]
                drug2pos[drug] = pos.split(',')
                drug2neg[drug] = neg.split(',')
                drug_set.add(drug)

        drug2smiles_file = '../datasets/STITCH_DDI/drug2smiles.tsv'
        drug2ecfp = {}
        with open(drug2smiles_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                drug, smiles = line[0], line[1]
                mol = MolFromSmiles(smiles)
                drug2ecfp[drug] = molecule2ecfp(mol, fp_radius=kwargs['fp_radius'], fp_length=kwargs['fp_length'])

                # # for debugging
                # drug2ecfp[drug] = np.array([1] * 1024)
        print('{} valid drugs'.format(len(drug2ecfp)))

        self.drug2ecfp = drug2ecfp
        self.drug2pos = drug2pos
        self.drug2neg = drug2neg
        self.drug_list = list(drug_set)
        return

    def __getitem__(self, index):
        drug = self.drug_list[index]
        pos = self.drug2pos[drug]
        neg = self.drug2neg[drug]
        pos = random.sample(pos, 1)[0]
        neg = random.sample(neg, 1)[0]

        drug_ecfp = self.drug2ecfp[drug]
        pos_ecfp = self.drug2ecfp[pos]
        neg_ecfp = self.drug2ecfp[neg]
        return drug_ecfp, pos_ecfp, neg_ecfp

    def __len__(self):
        return len(self.drug_list)

