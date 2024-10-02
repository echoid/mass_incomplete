@echo off

REM Define the datasets
set datasets=("hepatitis" "horse" "kidney" "mammo" "pima","winconsin")

REM Define the models
set models=("mice" "mean" "genrbf" "rbfn" "ppca" "ik" "kpca")
set models=("rbfn" "ik")

REM Loop through datasets
for %%d in %datasets% do (
    REM Loop through models
    for %%m in %models% do (
            REM Run the Python script with the current parameters
            python main.py --datasets %%~d --models %%~m --save
    )
)
