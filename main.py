import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from util import run_test  # Assuming this is your custom function
import os
import argparse
import matplotlib.pyplot as plt

# Generate synthetic datasets
syn_1 = dict(zip(["X", "Y"],
                 make_classification(n_samples=5000,
                                     n_features=20,
                                     n_informative=2,
                                     n_redundant=2,
                                     random_state=42)))
syn_2 = dict(zip(["X", "Y"],
                 make_classification(n_samples=1000,
                                     n_features=10,
                                     n_informative=5,
                                     n_redundant=5,
                                     random_state=42)))
syn_3 = dict(zip(["X", "Y"],
                 make_classification(n_samples=5000,
                                     n_features=20,
                                     n_informative=10,
                                     n_redundant=5,
                                     random_state=42)))

# Define the folder names and types
dataset_dict = {
    "syn_1": syn_1,
    "syn_2": syn_2,
    "syn_3": syn_3,
    "banknote": {"X": np.load("data/BNfeature.npy"), "Y": np.load("data/BNlabel.npy")},
    "genrbf": {"X": np.load("data/rbf/rbf_feature.npy"), "Y": np.load("data/rbf/rbf_label.npy")},
    "climate": {"X": np.load("data/climate_model_crashes/feature.npy"), "Y": np.load("data/climate_model_crashes/label.npy")},
    "concrete": {"X": np.load("data/concrete_compression/feature.npy"), "Y": np.load("data/concrete_compression/label.npy")},
    "wine": {"X": np.load("data/wine_quality_white/feature.npy"), "Y": np.load("data/wine_quality_white/label.npy")},
    "yacht": {"X": np.load("data/yacht_hydrodynamics/feature.npy"), "Y": np.load("data/yacht_hydrodynamics/label.npy")},
    #"yeast": {"X": np.load("data/yeast/feature.npy"), "Y": np.load("data/yeast/label.npy")},
    #"qsar": {"X": np.load("data/qsar_biodegradation/feature.npy"), "Y": np.load("data/qsar_biodegradation/label.npy")},
    "sonar": {"X": np.load("data/connectionist_bench_sonar/feature.npy"), "Y": np.load("data/connectionist_bench_sonar/label.npy")},
    "ionosphere": {"X": np.load("data/ionosphere/feature.npy"), "Y": np.load("data/ionosphere/label.npy")},
    
    }

def save_results_to_csv(dataname, typename, model, results):
    # Construct the directory path
    directory = f"result/{dataname}"
    
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
    
    # Construct the file path
    filename = f"{directory}/{typename}_{model}.csv"
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=["MissingRate", "F1_Score", "Accuracy"])
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run tests on datasets with specified models")
    parser.add_argument('--datasets', nargs='+', default=["syn_1", "syn_2", "syn_3", "banknote", "genrbf", "climate", "yeast", "qsar", "sonar","ionosphere"], help='List of datasets to run tests on')
    parser.add_argument('--models', nargs='+', default=['mass', 'mean', 'genrbf', 'mice'], help='List of models to run tests with')
    args = parser.parse_args()
    
    datasets = [dataset_dict[dataname] for dataname in args.datasets]
    datanames = args.datasets

    mymissrates = [None]
    mytypenames = ["default"]
    orimissrates = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    oritypenames = ['mnar', 'mcar']
    models = args.models

    for data, dataname in zip(datasets, datanames):
        if dataname == "genrbf":
            missrates = mymissrates
            typenames = mytypenames
        else:
            missrates = orimissrates
            typenames = oritypenames
        for typename in typenames:
            for model in models:
                results = []
                for missrate in missrates:
                    print(dataname, typename, model)
                    f1, acc = run_test(data, missrate, typename, model,dataname)
                    results.append((missrate, f1, acc))
                save_results_to_csv(dataname, typename, model, results)

if __name__ == "__main__":
    main()
