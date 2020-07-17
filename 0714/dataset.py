import scipy.io
import torch
from data import *
from utils import *
from utils import _get_explicit_property_prediction_node_feature, _get_property_prediction_node_feature, \
    _get_default_edge_feature, _get_node_dim, _get_edge_dim, _get_atom_distance, _get_atom_distance, \
    _get_max_atom_num_from_smiles_list, _get_max_atom_num_from_molecule_list, filter_out_invalid_smiles


class BBBPDataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super(BBBPDataset, self).__init__()
        self.model = kwargs['model']
        self.given_target = 'p_np'

        file_name = './datasets/BBBP.csv'
        smiles_list, task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        smiles_list, self.task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        print('smiles list: {}\tlabel list: {}'.format(len(smiles_list), len(self.task_label_list)))
        # print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class BACEDataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super(BACEDataset, self).__init__()
        self.model = kwargs['model']
        self.given_target = 'Class'

        file_name = './datasets/bace.csv'
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='mol', task_list_field=[self.given_target])
        # print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class DelaneyDataset(torch.utils.data.Dataset):
    given_target = 'measured log solubility in mols per litre'

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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class FreeSolvDataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super(FreeSolvDataset, self).__init__()
        self.model = kwargs['model']
        self.given_target = 'expt'

        file_name = './datasets/SAMPL.csv'
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        # print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class LipophilicityDataset(torch.utils.data.Dataset):

    def __init__(self, **kwargs):
        super(LipophilicityDataset, self).__init__()
        self.model = kwargs['model']
        self.given_target = 'exp'

        file_name = './datasets/Lipophilicity.csv'
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class CEPDataset(torch.utils.data.Dataset):
    given_target = 'PCE'

    def __init__(self, **kwargs):
        super(CEPDataset, self).__init__()
        self.model = kwargs['model']

        file_name = './datasets/cep-processed.csv'
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        # print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


def clean_up_molecule_label_list(molecule_list, label_list):
    a, b = [], []
    for mol in molecule_list:
        # Chem.SanitizeMol(mol)

        print(mol.GetProp('OpenBabel11221611003D'))
        try:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                temp = _get_explicit_property_prediction_node_feature(atom)
            a.append(mol)

        except:
            continue

    print(len(a), '\t', len(label_list))
    return molecule_list, label_list


class QM7Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(QM7Dataset, self).__init__()
        # TODO: debugging
        self.model = kwargs['model']
        self.given_target = 'u0_atom'

        sdf_file = './datasets/gdb7.sdf'
        csv_file = './datasets/gdb7.sdf.csv'
        molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
        _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=[self.given_target])
        # print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(molecule_list))) #7
        print('len of molecule\t', len(molecule_list), len(self.task_label_list)) # 7169, 7165

        clean_up_molecule_label_list(molecule_list, self.task_label_list)

        mat_file = './datasets/qm7.mat'
        dataset = scipy.io.loadmat(mat_file)
        print(list(dataset.keys()))

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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class QM7bDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(QM7bDataset, self).__init__()
        # TODO: debugging
        self.model = kwargs['model']
        self.given_target = 'u0_atom'

        sdf_file = './datasets/gdb7.sdf'
        csv_file = './datasets/gdb7.sdf.csv'
        molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
        _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=[self.given_target])
        # print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(molecule_list))) #7
        print('len of molecule\t', len(molecule_list), len(self.task_label_list)) # 7169, 7165

        clean_up_molecule_label_list(molecule_list, self.task_label_list)

        mat_file = './datasets/qm7.mat'
        dataset = scipy.io.loadmat(mat_file)
        print(list(dataset.keys()))

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
        return self.data[0][1].shape[-1]

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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


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
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class ZINC2MDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(ZINC2MDataset, self).__init__()
        self.model = kwargs['model']

        file_name = './datasets/dataset/zinc_standard_agent/processed/smiles.csv'
        self.smiles_list = from_smiles_csv(csv_file=file_name)
        print('max atom number: {}'.format(_get_max_atom_num_from_smiles_list(self.smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(self.smiles_list, **kwargs)

        return

    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            return ecfp
        else: # is graph
            node_feature, edge_feature, adjacency_list, distance_list = self.data[index]
            return node_feature, edge_feature, adjacency_list, distance_list

    @property
    def node_feature_dim(self):
        return self.data[0][0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[0][1].shape[-1]

    def __len__(self):
        return len(self.data)


class ChEMBLDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(ChEMBLDataset, self).__init__()

        self.model = kwargs['model']

        file_name = './datasets/dataset/chembl_filtered/processed/smiles.csv'
        self.smiles_list = from_smiles_csv(csv_file=file_name)
        print('max atom number: {}'.format(_get_max_atom_num_from_smiles_list(self.smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(self.smiles_list, **kwargs)

        return

    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            return ecfp
        else: # is graph
            node_feature, edge_feature, adjacency_list, distance_list = self.data[index]
            return node_feature, edge_feature, adjacency_list, distance_list

    @property
    def node_feature_dim(self):
        return self.data[0][0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[0][1].shape[-1]

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

