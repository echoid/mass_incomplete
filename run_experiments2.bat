@echo off


REM Check if CSV exists for mar/breast/rbfn.csv
if not exist "results\mar\breast\rbfn.csv" (
    echo Running experiment for dataset=breast, model=rbfn, missing_type=mar
    python main.py --datasets breast --models rbfn --missing_types mar --save
)

REM Check if CSV exists for mnar/breast/rbfn.csv
if not exist "results\mnar\breast\rbfn.csv" (
    echo Running experiment for dataset=breast, model=rbfn, missing_type=mnar
    python main.py --datasets breast --models rbfn --missing_types mnar --save
)

REM Check if CSV exists for mnar/australian/rbfn.csv
if not exist "results\mnar\australian\rbfn.csv" (
    echo Running experiment for dataset=australian, model=rbfn, missing_type=mnar
    python main.py --datasets australian --models rbfn --missing_types mnar --save
)

REM Check if CSV exists for mnar/heart/rbfn.csv
if not exist "results\mnar\heart\rbfn.csv" (
    echo Running experiment for dataset=heart, model=rbfn, missing_type=mnar
    python main.py --datasets heart --models rbfn --missing_types mnar --save
)


REM Check if CSV exists for mar/student/rbfn.csv
if not exist "results\mar\student\rbfn.csv" (
    echo Running experiment for dataset=student, model=rbfn, missing_type=mar
    python main.py --datasets student --models rbfn --missing_types mar --save
)

REM Check if CSV exists for mnar/student/rbfn.csv
if not exist "results\mnar\student\rbfn.csv" (
    echo Running experiment for dataset=student, model=rbfn, missing_type=mnar
    python main.py --datasets student --models rbfn --missing_types mnar --save
)
