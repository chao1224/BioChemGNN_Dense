# 1 STITCH

Refer to [this website](http://stitch.embl.de/cgi/download.pl?UserId=brZwE7NC7wm0&sessionId=NwiV0asmJEYj).

## 1.1 Drug

```
wget http://stitch.embl.de/download/chemicals.v5.0.tsv.gz
gunzip chemicals.v5.0.tsv.gz
```

## 1.2 Drug-Drug Interaction

```
mkdir -p STITCH_DDI
cd STITCH_DDI
wget http://stitch.embl.de/download/chemical_chemical.links.v5.0.tsv.gz
wget http://stitch.embl.de/download/chemical_chemical.links.detailed.v5.0.tsv.gz

gunzip chemical_chemical.links.v5.0.tsv.gz
gunzip chemical_chemical.links.detailed.v5.0.tsv.gz

cd ..
```

## 1.3 Drug-Protein Interaction

```
mkdir -p STRING_STITCH_DPI_PPI
cd STRING_STITCH_DPI_PPI

#wget http://stitch.embl.de/download/protein_chemical.links.v5.0.tsv.gz
#gunzip protein_chemical.links.v5.0.tsv.gz

wget http://stitch.embl.de/download/protein_chemical.links.v5.0/9606.protein_chemical.links.v5.0.tsv.gz
wget http://stitch.embl.de/download/protein_chemical.links.detailed.v5.0/9606.protein_chemical.links.detailed.v5.0.tsv.gz

gunzip 9606.protein_chemical.links.v5.0.tsv.gz
gunzip 9606.protein_chemical.links.detailed.v5.0.tsv.gz
cd ..
```

# 2 STRING

Refer to [this website](https://string-db.org/cgi/download.pl?sessionId=BROC69NXnwj6).

## 2.1 Protein

```
mkdir -p STRING_STITCH_DPI_PPI
cd STRING_STITCH_DPI_PPI

wget https://stringdb-static.org/download/protein.sequences.v11.0/9606.protein.sequences.v11.0.fa.gz
gunzip 9606.protein.sequences.v11.0.fa.gz

cd ..
```

## 2.2 Protein-Protein Interaction

```
cd STRING_STITCH_DPI_PPI

wget https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz
wget https://stringdb-static.org/download/protein.links.detailed.v11.0/9606.protein.links.detailed.v11.0.txt.gz

gunzip 9606.protein.links.v11.0.txt.gz
gunzip 9606.protein.links.detailed.v11.0.txt.gz

cd ..
```

# 3 Cheng's data (graph-based proximity)

Refer to [this website](https://www.nature.com/articles/s41467-019-09186-x).

```
mkdir -p Cheng-GP
cd Cheng-GP

wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM3_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM4_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM5_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM6_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM7_ESM.xlsx

cd ..
```

# 4 DrugBank

commercial?

# 5 ZINC250

Refer to [this git repo](https://github.com/chao1224/molecule_generation).

# 6 ChEMBL

Refer to [Graph Pre-Training git repo](https://github.com/snap-stanford/pretrain-gnns).

```
http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip

# adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
# first need to download the files and unzip:
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
# unzip and rename to chembl_with_labels
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
# into the dataPythonReduced directory
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl
```