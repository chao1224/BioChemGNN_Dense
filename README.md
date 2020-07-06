# Environment

```
conda create -n kg_mol python=3.7
source activate kg_mol

conda install ---yes -c pytorch pytorch=1.5 torchvision
conda install ---yes -c rdkit rdkit
conda install ---yes scikit-learn
conda install ---yes numpy
conda install ---yes  -c bioconda pubchempy
# pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```


# Experiments

| | ECFP | NEF | Weave | GG-NN | DTNN | enn-s2s | GIN | SchNet |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Delaney | Done | Done | | | Done | Ready | Done | Done |
| FreeSol | Done | Done | | | Done | Ready | Done | Done |
| Lipophilicity | Done | Done | | | Done | Ready | Done | Done |
| CEP | Done | Done | | | Done | Ready | Done | Done |
| QM8 | Done | Done | | | Done | Ready | Done | Done |
| QM9 | Done | Done | | | Done | Ready | Done | Done |
