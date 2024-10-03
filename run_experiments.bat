@echo off

@REM REM Define the datasets
@REM set datasets=("hepatitis" "horse" "kidney" "mammo" "pima","winconsin")
@REM set datasets=("car" "breast" "australian" "heart" 
@REM           "student" "banknote" "sonar" "spam" "wine")
@REM REM Define the models
@REM set models=("mice" "mean" "genrbf" "rbfn" "ppca" "ik" "kpca")
@REM set models=("em" "mean")
@REM REM Define missing types
@REM set missing_types=("mcar" "mnar" "mar")
@REM set missing_types=("mar")
@REM REM Loop through datasets

@REM for %%d in %datasets% do (
@REM     REM Loop through models
@REM     for %%m in %models% do (
@REM         REM Loop through missing types
@REM         for %%t in %missing_types% do (
@REM             REM Run the Python script with the current parameters
@REM             python main.py --datasets %%~d --models %%~m --missing_types %%~t --save
@REM         )
@REM     )
@REM )




@echo off

REM Define the datasets
set datasets="hepatitis horse kidney mammo pima winconsin"

REM Define the models
set models="mean mice em genrbf rbfn kpca ppca mpk"
set models="kpca ppca mpk"

REM Define missing types
set missing_types="default"
set missing_types="clustering"

REM Loop through datasets
for %%d in (%datasets%) do (
    REM Loop through models
    for %%m in (%models%) do (
        REM Loop through missing types
        for %%t in (%missing_types%) do (
            REM Run the Python script with the current parameters
            python main.py --datasets %%~d --models %%~m --missing_types %%~t --save --missing_rates 0.1 --cluster
        )
    )
)
