#!/bin/bash

# Define datasets
datasets=("wikitext-2" "alicewonderland" "rocstories")

# Define learning rates
lrs=("1e-3" "1e-4" "5e-5")

# Loop over datasets and learning rates
for dataset in "${datasets[@]}"; do
    for lr in "${lrs[@]}"; do
        echo "Running experiments for $dataset with lr=$lr ..."
        ./text-gen-env/bin/python3 main.py --dataset="$dataset" --model="lstm" --lr=$lr
        ./text-gen-env/bin/python3 main.py --dataset="$dataset" --model="transformer" --lr=$lr
        ./text-gen-env/bin/python3 main.py --dataset="$dataset" --model="xlstm" --lr=$lr
    done
done
