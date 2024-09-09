@echo off

REM Define the datasets
set datasets=("car" "breast" "australian" "heart" "student" "banknote" "sonar" "spam" "wine")

REM Define the models
set models=("mice" "mean" "genrbf")

REM Define the missing types
set missing_types=("mcar" "mar" "mnar")

REM Loop through datasets
for %%d in %datasets% do (
    REM Loop through models
    for %%m in %models% do (
        REM Loop through missing types
        for %%t in %missing_types% do (
            REM Run the Python script with the current parameters
            python main.py --datasets %%~d --models %%~m --missing_types %%~t --save
        )
    )
)
