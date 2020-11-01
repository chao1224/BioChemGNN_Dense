import pandas as pd


if __name__ == '__main__':
    # chembl_filtered
    chembl_filtered_file = './dataset/chembl_filtered/processed/smiles.csv'
    smiles_list = []
    with open(chembl_filtered_file, 'r') as f:
        for line in f:
            smiles_list.append(line.strip())

    print('{} SMILES in all'.format(len(smiles_list)))
    print('first ten {}'.format(smiles_list[:10]))
    print()

    # chembl_smiles2pubchemCid
    chembl_smiles2pubchemCid_file = './smiles2CID.txt'
    cid_list = []
    cid2smiles = {}
    with open(chembl_smiles2pubchemCid_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                cid = int(line[1])
                cid_list.append(cid)
                cid2smiles[cid] = line[0]
    print('{} valid CID, ChEMBL SMILES -> CID'.format(len(cid_list)))
    print()

    # # load chemical dictionary
    # stitch_file = '../chemicals.v5.0.tsv'
    # pubchem_stitch_cid_list = []
    # overlap = 0
    # with open(stitch_file, 'r') as f:
    #     next(f)
    #     for line in f:
    #         line = line.strip().split('\t')
    #         pubchem_stitch_cid_list.append(line[0])
    #         cid_ = line[0]
    #         if 'CIDs' in cid_: #?
    #             continue
    #         cid = int(line[0].replace('CIDs', '').replace('CIDm', ''))
    #         if cid in cid2smiles:
    #             overlap += 1
    #             if overlap<= 100:
    #                 print(line[-1], '\t', cid2smiles[cid])
    # print('{} drugs from STITCH'.format(len(pubchem_stitch_cid_list)))
    # print('{} overlaps between STITCH and ChEMBL'.format(overlap))
    # print()

    # load chemical dictionary
    stitch_file = '../STITCH_DDI/chemical_chemical.links.v5.0.tsv'
    pubchem_stitch_cid_set = set()
    with open(stitch_file, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            cid_1 = line[0]
            cid_2 = line[1]
            if 'CIDm' in cid_1:
                cid = int(cid_1.replace('CIDm', ''))
                pubchem_stitch_cid_set.add(cid)
            if 'CIDm' in cid_2:
                cid = int(cid_2.replace('CIDm', ''))
                pubchem_stitch_cid_set.add(cid)
    print('{} drugs from STITCH DDI'.format(len(pubchem_stitch_cid_set)))
    overlap = 0
    for cid in pubchem_stitch_cid_set:
        if cid in cid2smiles:
            overlap += 1
    print('{} overlaps between STITCH DDI'.format(overlap))
    print()

    string_file = '../STRING_STITCH_DPI_PPI/9606.protein_chemical.links.v5.0.tsv'
    pubchem_string_cid_set = set()
    overlap = 0
    with open(string_file, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            cid_1 = line[0]
            if 'CIDm' in cid_1:
                cid = int(cid_1.replace('CIDm', ''))
                pubchem_string_cid_set.add(cid)
    print('{} drugs from STRING'.format(len(pubchem_string_cid_set)))
    for cid in pubchem_string_cid_set:
        if cid in cid2smiles:
            overlap += 1
    print('{} overlaps between STITCH and ChEMBL'.format(overlap))
    print()

    overlap = 0
    for k,_ in cid2smiles.items():
        if k in pubchem_stitch_cid_set and k in pubchem_string_cid_set:
            overlap += 1
    print('{} overlaps among STRING, STITCH, ChEMBL'.format(overlap))
