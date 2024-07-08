#!/bin/bash

# Define datasets
datasets=("wikitext-2 alicewonderland rocstories")

# Define learning rates
lrs=("1e-4 5e-5")

# Loop over datasets and learning rates
for dataset in "${datasets[@]}"; do
    for lr in "${lrs[@]}"; do
        echo "Running experiments for $dataset with lr=$lr ..."
        /home/raquelpanadero/.pyenv/versions/3.11.0/envs/ml-env/bin/python3 main.py --dataset="$dataset" --model="lstm" --lr=$lr
        /home/raquelpanadero/.pyenv/versions/3.11.0/envs/ml-env/bin/python3 main.py --dataset="$dataset" --model="transformer" --lr=$lr
        /home/raquelpanadero/.pyenv/versions/3.11.0/envs/ml-env/bin/python3 main.py --dataset="$dataset" --model="xlstm" --lr=$lr
    done
done
