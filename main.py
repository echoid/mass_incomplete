import numpy as np
import pandas as pd
from utility import run
import os
import argparse



def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run tests on datasets with specified models")

    # Add argument for datasets with a set of options and a default
    parser.add_argument('--datasets', nargs='+', choices=["car", "breast", "australian", "heart", "adult", 
                                                          "student", "banknote", "sonar", "spam", "wine",
                                                          "hepatitis","horse","kidney","mammo","pima","winconsin"
                                                          ],
                        default=["banknote"],  # Set "banknote" as the default dataset
                        help='List of datasets to run tests on')

    # Add argument for models with a set of options and a default
    parser.add_argument('--models', nargs='+', choices=['mass', 'mean','mice', 
                                                        'genrbf','rbfn',
                                                        "ik","kpca","ppca",
                                                        "mpk","impk",
                                                        "mpk_KPCA","impk_KPCA"],
                        default=['mean'],  # Set "mean" as the default model
                        help='List of models to run tests with')

    # Add argument for missing types with a set of options and a default
    parser.add_argument('--missing_types', nargs='+', choices=["mcar", "mar", "mnar"],
                        default=["mcar"],  # Set "mcar" as the default missing type
                        help='List of missing data types to consider')
    
    # Add argument for missing rates with a set of options and a default
    parser.add_argument('--missing_rates', nargs='+', type=float, 
                        choices=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Example missing rates
                        default=None,  
                        help='List of missing data rates to consider')
    
    # Add a boolean flag for saving the results
    parser.add_argument('--save', action='store_true', help='Flag to save the results')

    # Parse the arguments
    args = parser.parse_args()

    datasets     = args.datasets
    models       = args.models
    if datasets[0] in ["hepatitis","horse","kidney","mammo","pima","winconsin"]:
        missing_types =  args.missing_types
        missing_rates = None
        clustering = True
        save = args.save
    else:
        missing_types = args.missing_types
        missing_rates = args.missing_rates
        save = args.save
        clustering = False
    
    for dataset in datasets:
        # Load data for the current dataset
        path = f"dataset/{dataset}/"
        #X = np.load(path + "features.npy")
        y = np.load(path + "label.npy")

        # Iterate through each model, missing type, and missing rate
        for missing_type in missing_types:
            for model in models:
                all_results = run(dataset, missing_type, model, missing_rates, y, clustering)
                
                if clustering:
                    result = pd.DataFrame(list(all_results.items()), columns=['Metric', 'Value'])
                else:
                    result = pd.DataFrame(all_results)

                output_path = f"results/{missing_type}/{dataset}/"
                os.makedirs(output_path, exist_ok=True)

                if save:
                    result.to_csv(os.path.join(output_path, f"{model}.csv"))
                    print(f"Results saved for dataset '{dataset}', missing type '{missing_type}', model '{model}'.")
                else:
                    print(f"Results not saved. Here are the results for dataset '{dataset}', missing type '{missing_type}', model '{model}':")
                    print(result)

if __name__ == "__main__":
    main()

