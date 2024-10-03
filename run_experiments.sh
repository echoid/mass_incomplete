#!/bin/bash


datasets=("hepatitis" "horse" "kidney" "mammo" "pima" "winconsin")

# List of models
models=("mean" "mice" "em" "genrbf" "rbfnn" "kpca" "ppca" "mpk")


# List of missing types
missing_types=("clf")



for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for missing_type in "${missing_types[@]}"; do
      python main.py --datasets $dataset --models $model --missing_types $missing_type --save
    done
  done
done