import random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles
import numpy as np
import pandas as pd

import torch

from utils import *
from utils import _get_explicit_property_prediction_node_feature, _get_property_prediction_node_feature, \
    _get_default_edge_feature, _get_node_dim, _get_edge_dim, _get_atom_distance, _get_atom_distance, \
    _get_max_atom_num_from_smiles_list


def from_3Dsdf(sdf_file, clean_mols=False):
    suppl = Chem.SDMolSupplier(sdf_file, clean_mols, False, False)
    molecule_list, target_list = [], defaultdict(list)
    for mol in suppl:
        molecule_list.append(mol)
    return molecule_list


def from_2Dcsv(csv_file, smiles_field, task_list_field):
    if smiles_field is not None:
        columns = [smiles_field] + task_list_field
        df = pd.read_csv(csv_file, usecols=columns)
        smiles_list = df[smiles_field].tolist()
    else:
        columns = task_list_field
        df = pd.read_csv(csv_file, usecols=columns)
        smiles_list = None
    task_label_list = []
    for task in task_list_field:
        task_label_list.append(df[task].tolist())
    task_label_list = np.stack(task_label_list, axis=1)
    return smiles_list, task_label_list


def transform(data_list, **kwargs):
    if kwargs['node_feature_func'] == 'property_prediction':
        node_feature_func = _get_property_prediction_node_feature
    elif kwargs['node_feature_func'] == 'explicit_property_prediction':
        node_feature_func = _get_explicit_property_prediction_node_feature
    node_feature_dim = _get_node_dim(node_feature_func)

    if kwargs['edge_feature_func'] == 'default':
        edge_feature_func = _get_default_edge_feature
    edge_feature_dim = _get_edge_dim(edge_feature_func)

    if kwargs['representation'] == 'molecule':
        if kwargs['model'] == 'ECFP':
            data_list = [molecule2ecfp(molecule,
                                       fp_radius=kwargs['fp_radius'],
                                       fp_length=kwargs['fp_length'])
                         for molecule in data_list]
        elif kwargs['model'] in ['NEF', 'enn-s2s', 'DTNN', 'GIN', 'SchNet']:
            data_list = [molecule2graph(molecule, max_atom_num=kwargs['max_atom_num'],
                                        node_feature_func=node_feature_func, node_feature_dim=node_feature_dim,
                                        edge_feature_func=edge_feature_func, edge_feature_dim=edge_feature_dim)
                         for molecule in data_list]
        else:
            raise ValueError('Model {} not included.'.format(kwargs['model']))
    elif kwargs['representation'] == 'smiles':
        if kwargs['model'] == 'ECFP':
            data_list = [smiles2ecfp(smiles,
                                     fp_radius=kwargs['fp_radius'],
                                     fp_length=kwargs['fp_length'])
                         for smiles in data_list]
        elif kwargs['model'] in ['NEF', 'enn-s2s', 'DTNN', 'GIN', 'SchNet']:
            data_list = [smiles2graph(smiles, max_atom_num=kwargs['max_atom_num'],
                                      node_feature_func=node_feature_func, node_feature_dim=node_feature_dim,
                                      edge_feature_func=edge_feature_func, edge_feature_dim=edge_feature_dim)
                         for smiles in data_list]
        else:
            raise ValueError('Model {} not included.'.format(kwargs['model']))
    else:
        raise ValueError('Representation {} not included.'.format(kwargs['representation']))
    return data_list


def molecule2graph(mol, max_atom_num, node_feature_func, node_feature_dim, edge_feature_func, edge_feature_dim, with_hydrogen=False, kekulize=False):
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)
    conformer = mol.GetConformers()[0]

    node_feature, edge_feature, adjacency, distance = extract_graph(
        mol=mol, conformer=conformer, max_atom_num=max_atom_num,
        node_feature_func=node_feature_func, node_feature_dim=node_feature_dim,
        edge_feature_func=edge_feature_func, edge_feature_dim=edge_feature_dim)

    return node_feature, edge_feature, adjacency, distance


def smiles2graph(smiles, max_atom_num, node_feature_func, node_feature_dim, edge_feature_func, edge_feature_dim, with_hydrogen=False, kekulize=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES `%s`' % smiles)
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformers()[0]

    node_feature, edge_feature, adjacency, distance = extract_graph(
        mol=mol, conformer=conformer, max_atom_num=max_atom_num,
        node_feature_func=node_feature_func, node_feature_dim=node_feature_dim,
        edge_feature_func=edge_feature_func, edge_feature_dim=edge_feature_dim)

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
    ECFP = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=fp_radius, nBits=fp_length)
    ECFP = np.array(list(ECFP.ToBitString()))
    ECFP = ECFP.astype(float)
    return ECFP


def smiles2ecfp(smiles, fp_radius, fp_length):
    molecule = Chem.MolFromSmiles(smiles)
    ECFP = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=fp_radius, nBits=fp_length)
    ECFP = np.array(list(ECFP.ToBitString()))
    ECFP = ECFP.astype(float)
    return ECFP
