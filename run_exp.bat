#!/bin/bash

# Run the commands to regenerate missing CSV files

echo "Rerunning: results/mnar/car/rbfn.csv"
python main.py --datasets car --models rbfn --missing_types mnar --save

echo "Rerunning: results/mar/breast/rbfn.csv"
python main.py --datasets breast --models rbfn --missing_types mar --save

echo "Rerunning: results/mnar/breast/rbfn.csv"
python main.py --datasets breast --models rbfn --missing_types mnar --save

echo "Rerunning: results/mnar/australian/rbfn.csv"
python main.py --datasets australian --models rbfn --missing_types mnar --save

echo "Rerunning: results/mnar/heart/rbfn.csv"
python main.py --datasets heart --models rbfn --missing_types mnar --save

echo "Rerunning: results/mar/student/rbfn.csv"
python main.py --datasets student --models rbfn --missing_types mar --save

echo "Rerunning: results/mnar/student/rbfn.csv"
python main.py --datasets student --models rbfn --missing_types mnar --save

echo "Rerunning: results/mnar/spam/rbfn.csv"
python main.py --datasets spam --models rbfn --missing_types mnar --save