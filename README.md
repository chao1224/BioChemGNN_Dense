# Environment

```
conda create -n kg_mol python=3.7
source activate kg_mol

conda install -y -c pytorch pytorch=1.5 torchvision
conda install -y -c rdkit rdkit
conda install -y scikit-learn
conda install -y numpy
conda install -y  -c bioconda pubchempy
# pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```


# Experiments

| | ECFP | NEF | Weave | GG-NN | DTNN | ENN-S2S | GIN | SchNet |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BBBP | Running | Running | | | Running | Running | Running | Running |
| BACE | Running | Running | | | Running | Running | Running | Running |
| Delaney | Done | Done | | | Done | Running | Done | Done |
| FreeSol | Done | Done | | | Done | Running | Done | Done |
| Lipophilicity | Done | Done | | | Done | Running | Done | Done |
| CEP | Done | Done | | | Done | Running | Done | Done |
| QM8 | Done | Done | | | Done | Running | Done | Done |
| QM9 | Done | Done | | | Done | Running | Done | Done |
