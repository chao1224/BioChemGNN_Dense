#!/bin/bash

#SBATCH --array=0-143%10
#SBATCH --output=log/%j_%a.out

source $HOME/.bashrc
conda activate drug
conda deactivate
conda activate drug

cd ../src

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

export task=$1
export model=$2
export seed=$3
export hyper_id=$SLURM_ARRAY_TASK_ID

export output_dir=../output/"$task"/"$model"/"$seed"
mkdir -p "$output_dir"

echo "$SLURM_JOB_ID"_"$SLURM_ARRAY_TASK_ID" > "$output_dir"/"$hyper_id".out
echo `date` >> "$output_dir"/"$hyper_id".out
python main.py --seed="$seed" $(cat ../hyper/"$task"/"$model"/"$hyper_id".hyper) >> "$output_dir"/"$hyper_id".out
echo `date` >> "$output_dir"/"$hyper_id".out
