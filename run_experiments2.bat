@echo off
setlocal enabledelayedexpansion

REM Define the datasets (without quotes)
set datasets=hepatitis horse kidney mammo pima winconsin

REM Define the models (without quotes)
set models=mean mice em genrbf rbfn kpca ppca

REM Define missing types (without quotes)
set missing_types=default

REM Loop through datasets
for %%d in (%datasets%) do (
    REM Loop through models
    for %%m in (%models%) do (
        REM Loop through missing types
        for %%t in (%missing_types%) do (
            REM Run the Python script with the current parameters
            python main.py --datasets %%d --models %%m --missing_types %%t --save --missing_rates 0.1 
        )
    )
)

endlocal
