from __future__ import print_function
import pickle
import os
import pandas as pd
from rdkit import Chem
from glob import glob
import re

# import deepchem as dc
# from deepchem.feat import rdkit_grid_featurizer as rgf
# from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates
# from deepchem.feat.graph_features import AtomicConvFeaturizer

def extract_labels(pdbbind_label_file):
    """Extract labels from pdbbind label file."""
    assert os.path.isfile(pdbbind_label_file)
    labels = {}
    with open(pdbbind_label_file) as f:
        content = f.readlines()
        for line in content:
            if line[0] == "#":
                continue
            line = line.split()
            # lines in the label file have format
            # PDB-code Resolution Release-Year -logKd Kd reference ligand-name
            #print line[0], line[3]
            labels[line[0]] = line[3]
    return labels


if __name__ == '__main__':
    pdb_stem_directory = './datasets/v2015'
    data_folder = './datasets/v2015'
    pdbbind_label_file = './datasets/v2015/INDEX_core_data.2013'
    pdbbind_label_file = './datasets/v2015/INDEX_general_PL_data.2015'

    labels = extract_labels(pdbbind_label_file)
    print('labels\t', len(labels))

    df_rows = []
    os.chdir(pdb_stem_directory)
    pdb_directories = [pdb.replace('/', '') for pdb in glob('*/')]
    print('len of pdb_directories\t', len(pdb_directories))

    # pdbs = pdb_directories
    #
    # protein_files = [os.path.join(data_folder, pdb, "%s_pocket.pdb" % pdb) for pdb in pdbs]
    # ligand_files = [os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs]
    #
    # # print(ligand_files)
    #
    # # for l_f in ligand_files:
    # #     suppl = Chem.SDMolSupplier(l_f, sanitize=False)
    # #     mol = suppl[0]
    # #     smiles = Chem.MolToSmiles(ligand_mol)
    # #     print(smiles)

    count = 0
    atoms_count = []
    for pdb_dir in pdb_directories:
        print("About to extract ligand and protein input files")
        pdb_id = os.path.basename(pdb_dir)
        # print('pdb_id', '\t', pdb_id, '\t', pdb_dir)
        # pdb_id = pdb_dir
        ligand_pdb = None
        protein_pdb = None
        # pdb_dir = '{}/{}'.format(pdb_stem_directory, pdb_dir)
        # print(pdb_dir, '\t', os.listdir(pdb_dir))
        for f in os.listdir(pdb_dir):
            if re.search("_ligand.sdf$", f):
                ligand_pdb = f
            elif re.search("_protein.pdb$", f):
                protein_pdb = f
            elif re.search("_ligand.mol2$", f):
                ligand_mol2 = f

        print("Extracted Input Files:")
        print(ligand_pdb, protein_pdb, ligand_mol2)
        if not ligand_pdb or not protein_pdb or not ligand_mol2:
            # raise ValueError("Required files not present for %s" % pdb_dir)
            continue
        ligand_pdb_path = os.path.join(pdb_dir, ligand_pdb)
        protein_pdb_path = os.path.join(pdb_dir, protein_pdb)
        ligand_mol2_path = os.path.join(pdb_dir, ligand_mol2)

        with open(protein_pdb_path, "rb") as f:
            protein_pdb_lines = f.readlines()

        with open(ligand_pdb_path, "rb") as f:
            ligand_pdb_lines = f.readlines()

        # try:
        #     with open(ligand_mol2_path, "rb") as f:
        #         ligand_mol2_lines = f.readlines()
        # except:
        #     ligand_mol2_lines = []
        ligand_mol2_lines = []

        try:
            print("About to compute ligand smiles string.")
            ligand_mol = Chem.SDMolSupplier(ligand_pdb_path, False)[0]
            # ligand_mol = Chem.MolFromPDBFile(ligand_pdb_path)
            if ligand_mol is None:
                continue
            smiles = Chem.MolToSmiles(ligand_mol)
            # print(smiles)
            complex_id = "%s%s" % (pdb_id, smiles)
            label = labels[pdb_id]
            # df_rows.append([pdb_id, smiles, complex_id, protein_pdb_lines,
            #                 ligand_pdb_lines, ligand_mol2_lines, label])
            atoms_count.append(ligand_mol.GetNumAtoms())
        except:
            continue

    print(len(atoms_count), '\t', min(atoms_count), max(atoms_count))
