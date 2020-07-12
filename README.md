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


# Supervised Experiments

| | ECFP | NEF | Weave | GG-NN | DTNN | ENN-S2S | GIN | SchNet |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BBBP | Running | Running | | | Running |  | Running | Running |
| BACE | Running | Running | | | Running |  | Running | Running |
| Delaney | Done | Done | | | Done | Running | Done | Done |
| FreeSol | Done | Done | | | Done | Running | Done | Done |
| Lipophilicity | Done | Done | | | Done | Running | Done | Done |
| CEP | Done | Done | | | Done | Running | Done | Done |
| QM8 | Done | Done | | | Done | Running | Done | Done |
| QM9 | Done | Done | | | Done | Running | Done | Done |

# Unsupervised Experiments

## Datasets

| Model | Dataset |
| :---: | :---: |
|[InfoGraph](https://arxiv.org/pdf/1908.01000.pdf) | QM9 (5k training, 10k validation, 10k testing, rest as unlabeled for supervised); random splitting |
|[Pre-training](https://arxiv.org/pdf/1905.12265.pdf)| 2m from ZINC15 for node-level pre-training; a preprocessed ChEMBL with 456K molecules and 1310 tasks; scaffold splitting; fine-tuning on BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV, BACE |
|[GROVER](https://arxiv.org/pdf/2007.02835.pdf) | 11m from ZINC15 and ChEMBL; RDKit for motif prediction;  |
| Minghao | same as pre-training |