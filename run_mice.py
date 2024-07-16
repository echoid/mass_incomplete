from sklearn.model_selection import train_test_split

import numpy as np
from sklearn import svm
import pandas as pd 
from mass import Modify_Kernel as MKernel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from mass import run_test



#pd.read_csv("for_echo/data/abalone_test.csv")
banknote = {"X":np.load("data/BNfeature.npy"), "Y":np.load("data/BNlabel.npy")}

abalone = pd.read_csv("data/abalone_test.csv",header = None)
abalone = {"X":np.array(abalone.iloc[:, :-1]),
            "Y":np.array(abalone.iloc[:, -1])}



# Step 1: Generate synthetic dataset
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


def save_results_to_csv(dataname, typename, model, results):
    filename = f"result/{dataname}_{typename}_{model}.csv"
    df = pd.DataFrame(results, columns=["MissingRate", "F1_Score", "Accuracy"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")



def main():
    datasets = [syn_1, syn_2, syn_3, banknote]
    datanames = ["syn_1", "syn_2", "syn_3", "banknote"]
    missrates = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    typenames = ['mnar']
    models = ["genRBF"]

    for data, dataname in zip(datasets, datanames):
        for typename in typenames:
            for model in models:
                results = []
                for missrate in missrates:
                    print(dataname, typename, model)
                    f1, acc = run_test(data, missrate, typename, model)
                    results.append((missrate, f1, acc))
                save_results_to_csv(dataname, typename, model, results)

if __name__ == "__main__":
    main()