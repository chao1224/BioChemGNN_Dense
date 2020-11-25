# BioChem Graph Neural Network --- Dense

This is the dense version of the Graph Neural Network for biology and chemistry molecules. The `dense` means each graph is padded to length of N, where N is the largest size of molecules in this dataset.

## Environment

```
conda create -n benchmark python=3.7
source activate benchmark

conda install -y -c pytorch pytorch=1.5 torchvision
conda install -y -c rdkit rdkit
conda install -y scikit-learn
conda install -y numpy
conda install -y matplotlib
```

## Models

| Model | Paper |
| :---: | :---: |
| ECFP | The generation of a unique machine description for chemical structures-a technique developed at chemical abstracts service, ACS Journal of Chemical Documentation 1965 |
| Neural Fingerprint (NEF) | Molecular graph convolutions: moving beyond fingerprints, NeurIPS 2016 |
| Graph Convolutional Network (GCN) | Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017 |
| Deep Tensor Neural Networks (DTNN) | Quantum-chemical insights from deep tensor neural networks, Nature Communications 2017 |
| Edge Neural Network (ENN) / MPNN | Neural Message Passing for Quantum Chemistry, ICML 2017 |
| SchNet | SchNet: A continuous-filter convolutional neural network for modeling quantum interactions, NeurIPS 2017 |
| Graph Isomorphism Network (GIN) | How Powerful are Graph Neural Networks?, ICLR 2019 |
| Directed Message Passing Neural Network (D-MPNN) | Analyzing Learned Molecular Representations for Property Prediction, ACS JCIM 2019 |

## Supervised Experiments

### random split

| | ECFP | NEF | DTNN | ENN | GCN | GIN | SchNet | D-MPNN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BACE | Done | Done | Done | Done | Done | Done | Done | Done |
| BBBP | Done | Done | Done | Done | Done | Done | Done | Done |
| Delaney | Done | Done | Done | Done | Done | Done | Done | Done |
| FreeSol | Done | Done | Done | Done | Done | Done | Done | Done |
