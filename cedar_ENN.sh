#!/usr/bin/env bash

cd 0626

export model=ENN


export epochs_list=(50 250)
export learning_rate_list=(0.001 0.003)
export task_list=(delaney freesolv lipophilicity cep qm8 qm9)
export running_index_list=(0 1 2 3 4)

export enn_hidden_dim_list=(32 64 128)
export enn_layer_num_list=(1 2 3 5)
export enn_fc_dim_list=(
"256"
"128 256 128"
"128 256"
"128"
)
export enn_readout_func_list=("set2set" "sum")
export enn_readout_func_list=("set2set")



for task in "${task_list[@]}"; do
  export count=0
  for epochs in "${epochs_list[@]}"; do
    for learning_rate in "${learning_rate_list[@]}"; do
      for enn_hidden_dim in "${enn_hidden_dim_list[@]}"; do
        for enn_layer_num in "${enn_layer_num_list[@]}"; do
          for enn_fc_dim in "${enn_fc_dim_list[@]}"; do
            for enn_readout_func in "${enn_readout_func_list[@]}"; do

                    for running_index in "${running_index_list[@]}"; do

                        export epochs="$epochs"
                        export learning_rate="$learning_rate"
                        export task="$task"
                        export running_index="$running_index"

                        export enn_hidden_dim="$enn_hidden_dim"
                        export enn_layer_num="$enn_layer_num"
                        export enn_fc_dim="$enn_fc_dim"
                        export enn_readout_func="$enn_readout_func"

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
                        --enn_hidden_dim="$enn_hidden_dim" \
                        --enn_layer_num="$enn_layer_num" \
                        --enn_fc_dim "$enn_fc_dim" \
                        --enn_readout_func="$enn_readout_func" \
                        --running_index="$running_index" \
                        --model_weight_dir="$model_weight_dir" --model_weight_path="$model_weight_path"

                    done
                    export count=$((count + 1))

            done
          done
        done
      done
    done
  done
done
