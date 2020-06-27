import csv
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import torch

from utils import *
from utils import _get_explicit_property_prediction_node_feature, _get_property_prediction_node_feature, \
    _get_default_edge_feature, _get_node_dim, _get_edge_dim, _get_atom_distance, _get_atom_distance, \
    _get_max_atom_num_from_smiles_list


def from_sdf(sdf_file, target_field, clean_mols=False):
    suppl = Chem.SDMolSupplier(sdf_file, clean_mols, False, False)
    molecule_list, target_list = [], defaultdict(list)
    for mol in suppl:
        molecule_list.append(mol)
    return molecule_list, target_list


def from_csv(csv_file, smiles_field, target_field):
    target_field = set(target_field)
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        fields = next(reader)
        smiles_list, target_list = [], defaultdict(list)
        for values in reader:
            if not any(values):
                continue
            for field, value in zip(fields, values):
                if field == smiles_field:
                    smiles_list.append(value)
                elif target_field is None or field in target_field:
                    target_list[field].append(eval(value))
    return smiles_list, target_list


def transform(data_list, **kwargs):
    if kwargs['task'] in ['qm7', 'qm8', 'qm9']:
        if kwargs['model'] == 'ECFP':
            data_list = [molecule2ecfp(molecule,
                                       fp_radius=kwargs['fp_radius'],
                                       fp_length=kwargs['fp_length'])
                         for molecule in data_list]
        elif kwargs['model'] in ['GIN', 'SchNet']:
            data_list = [molecule2graph(molecule,
                                      node_feature_func=kwargs['node_feature_func'],
                                      edge_feature_func=kwargs['edge_feature_func'],
                                      max_atom_num=kwargs['max_atom_num'])
                         for molecule in data_list]
            # node_feature, edge_feature, adjacency, distance = [], [], [], []
            # for molecule in data_list:
            #     node_feature_, edge_feature_, adjacency_, distance_ = molecule2graph(
            #         molecule,
            #         node_feature_func=kwargs['node_feature_func'],
            #         edge_feature_func=kwargs['edge_feature_func'],
            #         max_atom_num=kwargs['max_atom_num'])
            #     node_feature.append(node_feature_)
            #     edge_feature.append(edge_feature_)
            #     adjacency.append(adjacency_)
            #     distance.append(distance_)
            # node_feature = np.array(node_feature)
            # edge_feature = np.array(edge_feature)
            # adjacency = np.array(adjacency)
            # distance = np.array(distance)
            # data_list = [node_feature, edge_feature, adjacency, distance]
    else:
        if kwargs['model'] == 'ECFP':
            data_list = [smiles2ecfp(molecule,
                                     fp_radius=kwargs['fp_radius'],
                                     fp_length=kwargs['fp_length'])
                         for molecule in data_list]
        elif kwargs['model'] in ['GIN', 'SchNet']:
            data_list = [smiles2graph(molecule,
                                      node_feature_func=kwargs['node_feature_func'],
                                      edge_feature_func=kwargs['edge_feature_func'],
                                      max_atom_num=kwargs['max_atom_num'])
                         for molecule in data_list]
            # node_feature, edge_feature, adjacency, distance = [], [], [], []
            # for molecule in data_list:
            #     node_feature_, edge_feature_, adjacency_, distance_ = smiles2graph(
            #         molecule,
            #         node_feature_func=kwargs['node_feature_func'],
            #         edge_feature_func=kwargs['edge_feature_func'],
            #         max_atom_num=kwargs['max_atom_num'])
            #     node_feature.append(node_feature_)
            #     edge_feature.append(edge_feature_)
            #     adjacency.append(adjacency_)
            #     distance.append(distance_)
            # node_feature = np.array(node_feature)
            # edge_feature = np.array(edge_feature)
            # adjacency = np.array(adjacency)
            # distance = np.array(distance)
            # data_list = [node_feature, edge_feature, adjacency, distance]
    return data_list


def molecule2graph(mol, node_features, edge_features, max_atom_num, with_hydrogen=False, kekulize=False):
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)
    conformer = mol.GetConformers()[0]

    if node_feature_func == 'property_prediction':
        node_feature_func = _get_property_prediction_node_feature
    elif node_feature_func == 'explicit_property_prediction':
        node_feature_func = _get_explicit_property_prediction_node_feature

    if edge_feature_func == 'default':
        edge_feature_func = _get_default_edge_feature

    node_feature, edge_feature, adjacency, distance = extract_graph(mol, conformer, max_atom_num,
                                                                    node_feature_func, edge_feature_func)

    return node_feature, edge_feature, adjacency, distance


def smiles2graph(smiles, node_feature_func, edge_feature_func, max_atom_num, with_hydrogen=False, kekulize=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES `%s`' % smiles)
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformers()[0]

    if node_feature_func == 'property_prediction':
        node_feature_func = _get_property_prediction_node_feature
    elif node_feature_func == 'explicit_property_prediction':
        node_feature_func = _get_explicit_property_prediction_node_feature
    node_feature_dim = _get_node_dim(node_feature_func)

    if edge_feature_func == 'default':
        edge_feature_func = _get_default_edge_feature
    edge_feature_dim = _get_edge_dim(edge_feature_func)

    node_feature, edge_feature, adjacency, distance = extract_graph(mol, conformer, max_atom_num,
                                                                    node_feature_func, node_feature_dim,
                                                                    edge_feature_func, edge_feature_dim)

    return node_feature, edge_feature, adjacency, distance


def extract_graph(mol, conformer, max_atom_num, node_feature_func, node_feature_dim, edge_feature_func, edge_feature_dim):
    adjacency = np.zeros((max_atom_num, max_atom_num))
    adjacency = adjacency.astype(int)
    distance = np.zeros((max_atom_num, max_atom_num))

    node_feature = np.zeros((max_atom_num, node_feature_dim))
    node_feature = node_feature.astype(int)
    edge_feature = np.zeros((max_atom_num, max_atom_num, edge_feature_dim))
    edge_feature = edge_feature.astype(int)

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        node_feature[atom_idx] = node_feature_func(atom)

    for idx_i in range(mol.GetNumAtoms()):
        for idx_j in range(idx_i + 1, mol.GetNumAtoms()):
            dis = _get_atom_distance(conformer.GetAtomPosition(idx_i),
                                     conformer.GetAtomPosition(idx_j))
            distance[idx_i, idx_j] = dis
            distance[idx_j, idx_i] = dis

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        begin_index = begin_atom.GetIdx()
        end_index = end_atom.GetIdx()
        adjacency[begin_index, end_index] = 1
        adjacency[end_index, begin_index] = 1
        feat = edge_feature_func(bond)
        edge_feature[begin_index][end_index] = feat
        edge_feature[end_index][begin_index] = feat

    return node_feature, edge_feature, adjacency, distance


def molecule2ecfp(molecule, fp_radius, fp_length):
    ECFP = AllChem.GetMorganFingerprintAsBitVect(molecule, fp_radius, nBits=fp_length).ToBitString()
    ECFP = np.array(list(ECFP.ToBitString()))
    ECFP = ECFP.astype(float)
    return ECFP


def smiles2ecfp(smiles, fp_radius, fp_length):
    mol = Chem.MolFromSmiles(smiles)
    ECFP = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_length)
    ECFP = np.array(list(ECFP.ToBitString()))
    ECFP = ECFP.astype(float)
    return ECFP


class QM9Dataset(torch.utils.data.Dataset):
    target_field = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298']

    def __init__(self, **kwargs):
        super(QM9Dataset, self).__init__()
        self.model = kwargs['model']

        sdf_file = './datasets/qm9.sdf'
        csv_file = './datasets/qm9.sdf.csv'

        self.molecule_list, _ = from_sdf(sdf_file=sdf_file, target_field=None, clean_mols=False)
        _, self.target_list = from_csv(csv_file, smiles_field='', target_field=self.target_field)

        self.data = transform(self.molecule_list, **kwargs)

        return

    def __getitem__(self, index):
        item = {'graph': self.data[index]}
        item.update({k: v[index] for k, v in self.target_list.items()})
        return item

    @property
    def tasks(self):
        return list(self.target_list.keys())

    @property
    def node_feature_dim(self):
        return self.data[0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[1].shape[-1]

    def __len__(self):
        return len(self.data)


class DelaneyDataset(torch.utils.data.Dataset):
    given_target = 'measured log solubility in mols per litre'
    task = 'delaney'

    def __init__(self, **kwargs):
        super(DelaneyDataset, self).__init__()
        self.model = kwargs['model']

        file_name = './datasets/delaney-processed.csv'
        self.smiles_list, self.target_list = from_csv(csv_file=file_name, smiles_field='smiles', target_field=[self.given_target])
        print('max_atom_num:', _get_max_atom_num_from_smiles_list(self.smiles_list))
        self.target_list = self.target_list[self.given_target]
        self.data = transform(self.smiles_list, **kwargs)
        print(self.data[0][0].shape)

        return

    def __getitem__(self, index):
        if self.model == 'ECFP':
            ecfp = self.data[index]
            target = self.target_list[index]
            return ecfp, target
        else: # is graph
            node_feature, edge_feature, adjacency_list, distance_list = self.data[index]
            target = self.target_list[index]
            return node_feature, edge_feature, adjacency_list, distance_list, target

    @property
    def tasks(self):
        return list(self.target_list.keys())

    @property
    def node_feature_dim(self):
        return self.data[0][0].shape[-1]

    @property
    def edge_feature_dim(self):
        return self.data[1][0].shape[-1]

    def __len__(self):
        return len(self.data)