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
| Delaney | Done | Done | | | Running | | Done | Running |
| QM8 | Done | Running | | | Running | | Done | Running |
| QM9 | Done | Running | | | Running | | Done | Running |
