import scipy.io
import torch
from data import *
from utils import *
from utils import _get_explicit_property_prediction_node_feature, _get_property_prediction_node_feature, \
    _get_default_edge_feature, _get_node_dim, _get_edge_dim, _get_atom_distance, _get_atom_distance, \
    _get_max_atom_num_from_smiles_list, _get_max_atom_num_from_molecule_list, filter_out_invalid_smiles


class MoleculeDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            target = self.task_label_list[index]
            return ecfp, target
        else:
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


class BBBPDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(BBBPDataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'p_np'

        file_name = '{}/BBBP.csv'.format(self.root)
        smiles_list, task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        smiles_list, self.task_label_list = filter_out_invalid_smiles(smiles_list, task_label_list)
        print('smiles list: {}\tlabel list: {}'.format(len(smiles_list), len(self.task_label_list)))
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return


class BACEDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(BACEDataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'Class'

        file_name = '{}/bace.csv'.format(self.root)
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='mol', task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return


class DelaneyDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(DelaneyDataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'measured log solubility in mols per litre'

        file_name = '{}/delaney-processed.csv'.format(self.root)
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return


class FreeSolvDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(FreeSolvDataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'expt'

        file_name = '{}/SAMPL.csv'.format(self.root)
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return


class LipophilicityDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(LipophilicityDataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'exp'

        file_name = '{}/Lipophilicity.csv'.format(self.root)
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return


class CEPDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(CEPDataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'PCE'

        file_name = '{}/cep-processed.csv'.format(self.root)
        smiles_list, self.task_label_list = from_2Dcsv(csv_file=file_name, smiles_field='smiles', task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        kwargs['representation'] = 'smiles'
        self.data = transform(smiles_list, **kwargs)

        return


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


class QM7Dataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(QM7Dataset, self).__init__()
        # TODO: debugging
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'u0_atom'

        sdf_file = '{}/gdb7.sdf'.format(self.root)
        csv_file = '{}/gdb7.sdf.csv'.format(self.root)
        molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
        _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(molecule_list))) #7
        print('len of molecule\t', len(molecule_list), len(self.task_label_list)) # 7169, 7165

        clean_up_molecule_label_list(molecule_list, self.task_label_list)

        mat_file = './datasets/qm7.mat'
        dataset = scipy.io.loadmat(mat_file)
        print(list(dataset.keys()))

        kwargs['representation'] = 'molecule'
        self.data = transform(molecule_list, **kwargs)

        return


class QM7bDataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(QM7bDataset, self).__init__()
        # TODO: debugging
        self.root = kwargs['root']
        self.model = kwargs['model']
        self.given_target = 'u0_atom'

        sdf_file = '{}/gdb7.sdf'.format(self.root)
        csv_file = '{}/gdb7.sdf.csv'.format(self.root)

        molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
        _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=[self.given_target])
        print('max atom num: {}'.format(_get_max_atom_num_from_smiles_list(molecule_list))) #7
        print('len of molecule\t', len(molecule_list), len(self.task_label_list)) # 7169, 7165

        clean_up_molecule_label_list(molecule_list, self.task_label_list)

        mat_file = './datasets/qm7.mat'
        dataset = scipy.io.loadmat(mat_file)
        print(list(dataset.keys()))

        kwargs['representation'] = 'molecule'
        self.data = transform(molecule_list, **kwargs)

        return


class QM8Dataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(QM8Dataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        task_list = kwargs['task_list']

        if self.model == 'ECFP':
            csv_file = '{}/qm8.csv'.format(self.root)
            smiles_list, self.task_label_list = from_2Dcsv(csv_file, smiles_field='smiles', task_list_field=task_list)
            kwargs['representation'] = 'smiles'
            self.data = transform(smiles_list, **kwargs)
            print('max atom number: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
        else:
            sdf_file = '{}/qm8.sdf'.format(self.root)
            csv_file = '{}/qm8.sdf.csv'.format(self.root)
            molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
            _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=task_list)
            kwargs['representation'] = 'molecule'
            self.data = transform(molecule_list, **kwargs)
            print('max atom number: {}'.format(_get_max_atom_num_from_molecule_list(molecule_list)))

        return


class QM9Dataset(MoleculeDataset):
    def __init__(self, **kwargs):
        super(QM9Dataset, self).__init__()
        self.root = kwargs['root']
        self.model = kwargs['model']
        task_list = kwargs['task_list']

        if self.model == 'ECFP':
            csv_file = '{}/qm9.csv'.format(self.root)
            smiles_list, self.task_label_list = from_2Dcsv(csv_file, smiles_field='smiles', task_list_field=task_list)
            print('max atom number: {}'.format(_get_max_atom_num_from_smiles_list(smiles_list)))
            kwargs['representation'] = 'smiles'
            self.data = transform(smiles_list, **kwargs)
        else:
            sdf_file = '{}/qm9.sdf'.format(self.root)
            csv_file = '{}/qm9.sdf.csv'.format(self.root)
            molecule_list = from_3Dsdf(sdf_file=sdf_file, clean_mols=False)
            print('max atom number: {}'.format(_get_max_atom_num_from_molecule_list(molecule_list)))
            _, self.task_label_list = from_2Dcsv(csv_file, smiles_field=None, task_list_field=task_list)
            kwargs['representation'] = 'molecule'
            self.data = transform(molecule_list, **kwargs)

        return