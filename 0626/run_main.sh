#!/usr/bin/env bash

source $HOME/.bashrc
source activate kg_mol
source deactivate
source activate kg_mol

echo $@
date

echo "start"
python main.py $@
echo "end"
date
