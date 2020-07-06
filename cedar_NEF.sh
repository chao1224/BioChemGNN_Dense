#!/usr/bin/env bash

cd 0626

export model=NEF


export epochs_list=(100 1000)
export learning_rate_list=(0.001 0.003)
export task_list=(delaney freesolv lipophilicity cep qm8 qm9)
export running_index_list=(0 1 2 3 4)

nef_fp_length_list=(16 16 128 16 50 16 16 50 16 16 16 16 16 16 16 50 50 16 50 16 16 16 16 16 16 16 16 16 50 16 16 16 128 16 128 16 16 50 50 50 50 16 16 50 50)
nef_fp_hidden_dim_list=("512 128 64" "128 128 128 128" "20 20 20 20" "512 128 32" "512 512 512 512" "512 128 64" "128 128 128 128" "128 128 128 128" "128 128 128 128" "512 128 32" "20 20 20 20" "512 128 64" "512 128 64" "512 512 512 512" "512 128 64" "512 512 512 512" "512 128 64" "512 128 32" "512 128 64" "512 128 64" "20 20 20 20" "512 128 64" "512 128 32" "512 128 32" "128 128 128 128" "128 128 128 128" "20 20 20 20" "128 128 128 128" "512 128 32" "512 128 32" "128 128 128 128" "20 20 20 20" "512 128 32" "512 128 32" "512 128 64" "512 128 64" "512 512 512 512" "512 128 64" "512 128 32" "512 128 32" "512 128 64" "512 512 512 512" "512 128 64" "512 128 64" "128 128 128 128")
nef_fc_hidden_dim_list=("512 128 32" "32 4" "32 4" "512 128" "128 8" "256 64 16" "512 128 32" "64 8" "64 8" "512 128 32" "512 128 32" "16" "128 16" "512 128 32" "32 4" "128 16" "32" "64" "32 4" "256 64" "16" "32" "128 8" "256 64" "256 64" "128 16" "32 4" "256 64 16" "64 8" "128 16" "32" "128 8" "32 4" "256 64 16" "32 4" "128" "32 4" "64 8" "64" "256 64 16" "64" "256 64 16" "64 8" "512" "256 64 16")


for task in "${task_list[@]}"; do
  export count=0
  for epochs in "${epochs_list[@]}"; do
    for learning_rate in "${learning_rate_list[@]}"; do
      for idx in {0..44}; do

                  for running_index in "${running_index_list[@]}"; do

                    export epochs="$epochs"
                    export learning_rate="$learning_rate"
                    export task="$task"
                    export nef_fp_length="${nef_fp_length_list[$idx]}"
                    export nef_fp_hidden_dim="${nef_fp_hidden_dim_list[$idx]}"
                    export nef_fc_hiddden_dim="${nef_fc_hidden_dim_list[$idx]}"

                    export running_index="$running_index"

                    export output_dir=../output/"$model"/"$task"
                    export output_path="$output_dir"/"$count"_"$running_index".out

                    export model_weight_dir=../model_weight/"$model"/"$task"
                    export model_weight_path="$model_weight_dir"/"$count"_"$running_index".pt

                    mkdir -p "$output_dir"
                    mkdir -p "$model_weight_dir"

                    sbatch --gres=gpu:v100l:1 -c 4 --mem=30G -t 6:00:00  --account=rrg-bengioy-ad --qos=high --job-name="$model"_"$task" \
                    --output="$output_path" \
                    ./run_main.sh \
                    --model="$model" \
                    --epochs="$epochs" --learning_rate="$learning_rate" \
                    --task="$task" \
                    --nef_fp_length="$nef_fp_length" \
                    --nef_fp_hidden_dim "$nef_fp_hidden_dim" \
                    --nef_fc_hiddden_dim "$nef_fc_hiddden_dim" \
                    --running_index="$running_index" \
                    --model_weight_dir="$model_weight_dir" --model_weight_path="$model_weight_path"

                  done
                  export count=$((count + 1))

      done
    done
  done
done






#export nef_fp_length_list=(
#50
#16
#128
#1024
#)
#
#export nef_fp_hidden_dim_list=(
#"20 20 20 20"
#"128 128 128 128"
#"512 512 512 512"
#"512 128 64"
#"512 128 32"
#)
#
#export nef_fc_hiddden_dim_list=(
#"128 8"
#"512"
#"512 128"
#"512 128 32"
#"256"
#"256 64"
#"256 64 16"
#"128"
#"128 16"
#"64"
#"64 8"
#"32"
#"32 4"
#"16")

