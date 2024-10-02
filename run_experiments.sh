#!/bin/bash

# List of datasets
datasets=("car" "breast" "australian" "heart" "adult" 
          "student" "banknote" "sonar" "spam" "wine")

# List of models
models=("mean" "mice")

# List of missing types
missing_types=("mcar" "mar" "mnar")



# List of missing rates
#missing_rates=("0.05" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8")

# Iterate over each combination of dataset, model, missing type, and missing rate
# for dataset in "${datasets[@]}"; do
#   for model in "${models[@]}"; do
#     for missing_type in "${missing_types[@]}"; do
#       for missing_rate in "${missing_rates[@]}"; do
#         # Run the Python script with the specified arguments
#         python main.py --datasets $dataset --models $model --missing_types $missing_type --missing_rates $missing_rate --save True
#       done
#     done
#   done
# done


# List of datasets
datasets=("car" "breast" "australian" "heart" "adult" 
          "student" "banknote" "sonar" "spam" "wine")

datasets=("hepatitis" "horse" "kidney" "mammo" "pima" "winconsin")

datasets=("car" "breast" "heart" "sonar")

datasets=("car" "breast" "australian" "heart"  
          "student" "banknote" "sonar" "spam" "wine")

# List of models
models=("em")


# List of missing types
missing_types=("mcar" "mnar" "mar")


# Car: genrbf, 
# Breast, genrbf, ik, ppca_0
# Heart, genrbf, ik , ppca_0

# for dataset in "${datasets[@]}"; do
#   for model in "${models[@]}"; do
#     for missing_type in "${missing_types[@]}"; do
#       python main.py --datasets $dataset --models $model --missing_types $missing_type --save
#     done
#   done
# done

#echo "All experiments have been executed."


for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for missing_type in "${missing_types[@]}"; do
      python main.py --datasets $dataset --models $model --missing_types $missing_type --save
    done
  done
done