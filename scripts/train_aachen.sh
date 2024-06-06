#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("aachen")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/aachen"

mkdir -p "$out_dir"

for scene in ${scenes[*]}; do
  torchrun --standalone --nnodes 1 --nproc-per-node 8 $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" \
        --num_head_blocks 3 --mlp_ratio 2 --num_decoder_clusters 50 --max_iterations 100000  
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" -hyps 3200 2>&1 | tee "$out_dir/log_${scene}.txt"
done

for scene in ${scenes[*]}; do
  echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -2 | head -1)"
done
