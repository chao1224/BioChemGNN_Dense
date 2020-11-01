## DDI on STITCH
mkdir -p STITCH_DDI
cd STITCH_DDI
wget http://stitch.embl.de/download/chemical_chemical.links.v5.0.tsv.gz
wget http://stitch.embl.de/download/chemical_chemical.links.detailed.v5.0.tsv.gz

gunzip chemical_chemical.links.v5.0.tsv.gz
gunzip chemical_chemical.links.detailed.v5.0.tsv.gz

cd ..


## DPI on STITCH
mkdir -p STRING_STITCH_DPI_PPI
cd STRING_STITCH_DPI_PPI

#wget http://stitch.embl.de/download/protein_chemical.links.v5.0.tsv.gz
#gunzip protein_chemical.links.v5.0.tsv.gz

wget http://stitch.embl.de/download/protein_chemical.links.v5.0/9606.protein_chemical.links.v5.0.tsv.gz
wget http://stitch.embl.de/download/protein_chemical.links.detailed.v5.0/9606.protein_chemical.links.detailed.v5.0.tsv.gz

gunzip 9606.protein_chemical.links.v5.0.tsv.gz
gunzip 9606.protein_chemical.links.detailed.v5.0.tsv.gz
cd ..


## protein on STRING
mkdir -p STRING_STITCH_DPI_PPI
cd STRING_STITCH_DPI_PPI

wget https://stringdb-static.org/download/protein.sequences.v11.0/9606.protein.sequences.v11.0.fa.gz
gunzip 9606.protein.sequences.v11.0.fa.gz

cd ..


## PPI on STRING
cd STRING_STITCH_DPI_PPI

wget https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz
wget https://stringdb-static.org/download/protein.links.detailed.v11.0/9606.protein.links.detailed.v11.0.txt.gz

gunzip 9606.protein.links.v11.0.txt.gz
gunzip 9606.protein.links.detailed.v11.0.txt.gz

cd ..



## Cheng GP
mkdir -p Cheng-GP
cd Cheng-GP

wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM3_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM4_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM5_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM6_ESM.xlsx
wget https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09186-x/MediaObjects/41467_2019_9186_MOESM7_ESM.xlsx

cd ..

