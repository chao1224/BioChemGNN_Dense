import numpy as np
import torch
from rdkit import Chem
from sklearn.model_selection import KFold, StratifiedKFold
import math

atom_candidates = ['C', 'Cl', 'I', 'F', 'O', 'N', 'P', 'S', 'Br', 'Unknown']


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: 1 if x == s else 0, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        logger.info('Unknown detected: {}'.format(x))
        x = allowable_set[-1]
    return list(map(lambda s: 1 if x == s else 0, allowable_set))


def _get_atom_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


def _get_max_atom_num_from_smiles_list(smiles_list):
    molecule_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    num_list = [mol.GetNumAtoms() for mol in molecule_list]
    return max(num_list)


def _get_max_atom_num_from_molecule_list(molecule_list):
    num_list = [mol.GetNumAtoms() for mol in molecule_list]
    return max(num_list)


def _get_explicit_property_prediction_node_feature(atom):
    return np.array([
        one_of_k_encoding_unk(atom.GetSymbol(), atom_candidates) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3]) +
        one_of_k_encoding(atom.GetIsAromatic(), [0, 1])
    ])


def _get_property_prediction_node_feature(atom):
    return np.array([
        one_of_k_encoding_unk(atom.GetSymbol(), atom_candidates) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3]) +
        one_of_k_encoding(atom.GetIsAromatic(), [0, 1])
    ])


def _get_node_dim(node_feature_func):
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(node_feature_func(a)[0])


def _get_default_edge_feature(bond):
    bt = bond.GetBondType()
    bond_features = np.array([bt == Chem.rdchem.BondType.SINGLE,
                              bt == Chem.rdchem.BondType.DOUBLE,
                              bt == Chem.rdchem.BondType.TRIPLE,
                              bt == Chem.rdchem.BondType.AROMATIC,
                              bond.GetIsConjugated(),
                              bond.IsInRing()])
    bond_features = bond_features.astype(int)
    return bond_features


def _get_edge_dim(edge_feature_func):
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(edge_feature_func(simple_mol.GetBonds()[0]))


def split_into_KFold(dataset, k, index, **kwargs):
    indices_list = []
    if kwargs['k_fold'] == 'StratifiedKFold':
        pass
    elif kwargs['k_fold'] == 'KFold':
        kf = KFold(n_splits=k, shuffle=True, random_state=kwargs['seed'])
        for _, idx in kf.split(np.arange(len(dataset))):
            indices_list.append(idx)
    else:
        raise ValueError

    test_indices = indices_list[index]
    indices = np.ones(len(dataset), dtype=bool)
    indices[test_indices] = 0
    train_indices = indices.nonzero()[0]
    
    return train_indices, test_indices


def area_under_roc(pred, target):
    order = pred.argsort(descending=True)
    target = target[order]
    hit = target.cumsum(0)
    all = (target == 0).sum() * (target == 1).sum()
    auroc = hit[target == 0].sum() / (all + 1e-10)

    return auroc


def area_under_prc(pred, target):
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(len(target))
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)

    return auprc


def root_mean_squared_error(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mean_absolute_error(pred, target):
    return torch.mean(torch.abs(pred - target))


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()