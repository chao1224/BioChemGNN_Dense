from __future__ import print_function
import pickle
import os
import pandas as pd
from rdkit import Chem
from glob import glob
import re

from data import *
from utils import *
from utils import _get_explicit_property_prediction_node_feature, _get_property_prediction_node_feature, \
    _get_default_edge_feature, _get_node_dim, _get_edge_dim, _get_atom_distance, _get_atom_distance, \
    _get_max_atom_num_from_smiles_list, _get_max_atom_num_from_molecule_list



if __name__ == '__main__':
    sdf_file = 'datasets/gdb7.sdf'

    molecules_list = Chem.SDMolSupplier(sdf_file, False)
    atoms_count = []

    for molecule in molecules_list:
        # try:
        #     if molecule is None:
        #         print('none')
        #         continue

            for atom in molecule.GetAtoms():
                atom_idx = atom.GetIdx()
                temp = _get_explicit_property_prediction_node_feature(atom)
            smiles = Chem.MolToSmiles(molecule)
            # print(smiles)
            atoms_count.append(molecule.GetNumAtoms())
        # except:
        #     continue

    print(len(atoms_count), '\t', min(atoms_count), max(atoms_count))
