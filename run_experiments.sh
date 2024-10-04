#!/bin/bash


datasets=("hepatitis" "horse" "kidney" "mammo" "pima" "winconsin")

# List of models
models=("mean" "mice" "em" "genrbf" "rbfnn" "kpca" "ppca" "mpk")
models=("impk" "mpk")
models=("gower" "simple")


# List of missing types
missing_types=("clustering")


for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for missing_type in "${missing_types[@]}"; do
      python main.py --datasets $dataset --models $model --missing_types clustering --missing_rates 0.1 --save --cluster  
    done
  done
done