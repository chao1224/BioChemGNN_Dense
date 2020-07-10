#!/usr/bin/env bash

cd 0626

export model=DTNN


export epochs_list=(50 500)
export learning_rate_list=(0.001 0.003)
export task_list=(delaney freesolv lipophilicity cep qm8 qm9)
export task_list=(bace bbbp)
export running_index_list=(0 1 2 3 4)

export dtnn_hidden_dim_list=("64 64 64" "64 32" "32 32 32" "32 16" "16 16" "16")
export dtnn_fc_hidden_dim_list=("128" "128 8" "64" "16" "8" "")

for task in "${task_list[@]}"; do
  export count=0
  for epochs in "${epochs_list[@]}"; do
    for learning_rate in "${learning_rate_list[@]}"; do
      for dtnn_hidden_dim in "${dtnn_hidden_dim_list[@]}"; do
        for dtnn_fc_hidden_dim in "${dtnn_fc_hidden_dim_list[@]}"; do
            for running_index in "${running_index_list[@]}"; do

                export epochs="$epochs"
                export learning_rate="$learning_rate"
                export task="$task"
                export running_index="$running_index"

                export dtnn_hidden_dim="$dtnn_hidden_dim"
                export dtnn_fc_hidden_dim="$dtnn_fc_hidden_dim"

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
                --dtnn_hidden_dim "$dtnn_hidden_dim" \
                --dtnn_fc_hidden_dim "$dtnn_fc_hidden_dim" \
                --running_index="$running_index" \
                --model_weight_dir="$model_weight_dir" --model_weight_path="$model_weight_path"

            done
            export count=$((count + 1))

        done
      done
    done
  done
done
