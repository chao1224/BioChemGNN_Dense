#!/usr/bin/env bash

source $HOME/.bashrc
source activate kg_mol
source deactivate
source activate kg_mol

echo $@
date

echo "start"
python pre_train.py $@
echo "end"
date
