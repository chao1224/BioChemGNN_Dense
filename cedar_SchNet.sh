#!/usr/bin/env bash

cd 0626

export model=SchNet


export epochs_list=(50 500)
export learning_rate_list=(0.001 0.003)
export task_list=(delaney qm8 qm9)
export running_index_list=(0 1 2 3 4)


for task in "${task_list[@]}"; do
  export count=0
  for epochs in "${epochs_list[@]}"; do
    for learning_rate in "${learning_rate_list[@]}"; do
      for running_index in "${running_index_list[@]}"; do

          export epochs="$epochs"
          export learning_rate="$learning_rate"
          export task="$task"
          export running_index="$running_index"

          export output_dir=../output/"$model"/"$task"
          export output_path="$output_dir"/"$count"_"$running_index".out

          export model_weight_dir=../model_weight/"$model"/"$task"
          export model_weight_path="$model_weight_dir"/"$count"_"$running_index".pt

          mkdir -p "$output_dir"
          mkdir -p "$model_weight_dir"

          sbatch --gres=gpu:v100l:1 -c 4 --mem=30G -t 3:00:00  --account=rrg-bengioy-ad --qos=high --job-name="$model"_"$task" \
          --output="$output_path" \
          ./run_main.sh \
          --model="$model" \
          --epochs="$epochs" --learning_rate="$learning_rate" \
          --task="$task" \
          --running_index="$running_index" \
          --model_weight_dir="$model_weight_dir" --model_weight_path="$model_weight_path"

      done
      export count=$((count + 1))
    done
  done
done
