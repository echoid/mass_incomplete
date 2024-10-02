param_value = None
data_stats = None
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.decomposition import KernelPCA
#from genrbf.run_genrbf import run_genrbf
#from rbfn_model import run_rbfn
from tqdm import tqdm
#from ik import Isolation_Kernal,run_ppca,run_kpca
from mass_model import run_mpk, run_impk
# from sklearn.svm import SVC
# from sklearn.decomposition import KernelPCA
# from mass import Modify_Kernel as MKernel



from sklearn.model_selection import StratifiedKFold

def stats_convert(column_info):
    if all(column == "numerical" for column in column_info.values()):
        return None
    stats = {"attribute": []}
    for column_name, column in column_info.items():
        col_dic = {'type': ''}
        if column == "numerical":
            col_dic['type'] = "Numeric"
        elif isinstance(column, dict):
            key = next(iter(column))
            if key in {"ordinal", "nominal"}:
                col_dic['type'] = key.capitalize()
                col_dic['values'] = list(column[key].values())
        stats["attribute"].append(col_dic)

    return stats

def run(dataset, missing_type, model, missing_rates, y):
    
    na_path = f"dataset_nan/{dataset}/{missing_type}/"

    if missing_rates is None: 
        missing_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    if model in ["mpk","impk"]:
        with open(f"dataset/{dataset}/column_info.json", 'r') as file:
            column_info = json.load(file)
            data_stats = stats_convert(column_info)
    else:
        data_stats = None

    all_results = {}

    for rate in tqdm(missing_rates):
        data_na = np.load(na_path + f"{rate}.npy")
        
        skf = StratifiedKFold(n_splits=5)
        results_list = []

        for id_acc, (trn_index, test_index) in enumerate(skf.split(data_na, y)):
            
            X_train, X_test = data_na[trn_index], data_na[test_index]
            y_train, y_test = y[trn_index], y[test_index]
            
            # Run the model and get the evaluation results
            results = run_model(model, X_train, X_test, y_train, y_test, data_stats)
            results_list.append(results)
        # Aggregate results for this missing rate
        all_results[rate] = aggregate_results(results_list)

    return all_results





def run_model(model, X_train, X_test, y_train, y_test,data_stats):
    if model == "mean":
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        results = SVC_evaluation(X_train, y_train, X_test, y_test)

    elif model == "mice":
        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        results = SVC_evaluation(X_train, y_train, X_test, y_test)

    elif model == "genrbf":
        train,test = run_genrbf(X_train.astype(np.float64), X_test.astype(np.float64))
        results = SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")
    elif model == "rbfn":
        results = run_rbfn(X_train, X_test, y_train, y_test)

    elif model == "ppca":
        print("PPCA + MICE")
        # PPCA + MICE
        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        train,test = run_ppca(X_train,X_test)
        results = SVC_evaluation(train, y_train, test, y_test)

    elif model == "kpca":
        print("KPCA + MICE")
        #KPCA + MICE
        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        train,test = run_kpca(X_train,X_test)
        results = SVC_evaluation(train, y_train, test, y_test)

    elif model == "ik":
        print("IK + MICE")
        # IK + MICE
        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        IK = Isolation_Kernal(psi=128, t=200, KD_tree=True)
        IK.build(X_train)
        train_feature = IK.generate_feature(X_train)
        test_feature = IK.generate_feature(X_test)
        test_sim = IK.similarity(test_feature,train_feature)
        train_sim = IK.similarity(train_feature,train_feature)
        results = SVC_evaluation(train_sim, y_train, test_sim, y_test, kernel="precomputed")


    elif model == "mpk":
        print("MPK + MICE")
        # MPK + MICE/MODE
        #X_train, X_test, y_train, y_test = sampling(X_train, X_test, y_train, y_test)
        X_train, X_test = mice_mode_imputer(X_train, X_test, data_stats)

        train, test  = run_mpk(X_train, X_test, data_stats)
        
        results = SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")

    elif model == "impk":
        print("iMPK")
        X_train, X_test, y_train, y_test = sampling(X_train, X_test, y_train, y_test)
        train, test  = run_impk(X_train, X_test, data_stats)

        results = SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")


    elif model == "mpk_KPCA":
        print("MPK + MICE + KPCA")
        # MPK + MICE/MODE
        #X_train, X_test, y_train, y_test = sampling(X_train, X_test, y_train, y_test)
        X_train, X_test = mice_mode_imputer(X_train, X_test, data_stats)
        train, test  = run_mpk(X_train, X_test, data_stats)
        try:
            train, test  = KernelPCA_with_precomputed(train,test)
            results = SVC_evaluation(train, y_train, test, y_test, kernel="linear")
        except:
            results = SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")

    elif model == "impk_KPCA":
        print("iMPK + KPCA")
        #X_train, X_test, y_train, y_test = sampling(X_train, X_test, y_train, y_test)
        train, test  = run_impk(X_train, X_test, data_stats)
        try:
            train, test  = KernelPCA_with_precomputed(train,test)
            results = SVC_evaluation(train, y_train, test, y_test, kernel="linear")
        except:
            results = SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")
    return results

def KernelPCA_with_precomputed(train,test):
    # Perform KernelPCA using the precomputed kernel matrix
    kpca = KernelPCA(kernel='precomputed')
    train = kpca.fit_transform(train)
    test = kpca.transform(test)
    
    return train, test

def SVC_evaluation(X_train, y_train, X_test, y_test, kernel="rbf"):

    # Check for NaN values in the input data
    if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
        return {
            "accuracy": np.nan,
            "f1_score": np.nan,
        }
    C = 1

    # Define the SVC model
    svc = SVC(C=C, kernel=kernel)

    # Train the model on the training data
    svc.fit(X_train, y_train)

    # Evaluate on the test data
    y_pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    results = {
        "accuracy": accuracy,
        "f1_score": f1,
    }

    return results

def aggregate_results(results_list):
    avg_accuracy = np.mean([result['accuracy'] for result in results_list])
    avg_f1_score = np.mean([result['f1_score'] for result in results_list])
    std_accuracy = np.std([result['accuracy'] for result in results_list])
    std_f1_score = np.std([result['f1_score'] for result in results_list])

    return {
        "avg_accuracy": avg_accuracy,
        "std_accuracy": std_accuracy,
        "avg_f1_score": avg_f1_score,
        "std_f1_score": std_f1_score
    }

def sampling(X_train, X_test, y_train, y_test, sample_size=0.1, random_state=42):

    # Perform sampling on X_train and y_train
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, test_size=1 - sample_size, random_state=random_state
    )

    # Perform sampling on X_test and y_test
    X_test_sample, _, y_test_sample, _ = train_test_split(
        X_test, y_test, test_size=1 - sample_size, random_state=random_state
    )

    return X_train_sample, X_test_sample, y_train_sample, y_test_sample
def mice_mode_imputer(X_train, X_test, data_stats):

    # Define imputers
    numerical_imputer = IterativeImputer()          # For numerical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')  # For categorical columns

    # Iterate over each column based on data_stats
    if data_stats:
        for i, attr in enumerate(data_stats['attribute']):
            col_type = attr['type']
            
            # Check the type of the column and apply the appropriate imputer
            if col_type == 'Numeric':
                X_train[:, i:i+1] = numerical_imputer.fit_transform(X_train[:, i:i+1])
                X_test[:, i:i+1] = numerical_imputer.transform(X_test[:, i:i+1])
            else:
                # Impute categorical columns using SimpleImputer with 'most_frequent' strategy
                X_train[:, i:i+1] = categorical_imputer.fit_transform(X_train[:, i:i+1])
                X_test[:, i:i+1] = categorical_imputer.transform(X_test[:, i:i+1])
    else:
        imputer = IterativeImputer()
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    return X_train, X_test




# def run_test(data,missing_rate,mtype = "mcar",model = "mass",data_stats = None):
#     X = data["X"]
#     Y = data["Y"]
#     train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


#     train_X = make_missing(train_X,rate = missing_rate,type = mtype)
#     test_X = make_missing(test_X,rate = missing_rate,type = mtype)
    
#     if model == "mass":
#         kernal = MKernel(None, data_stats)
#         kernal.set_nbins(param_value)


#         train_mod, test_mod = kernal.build_model(train_X, test_X)  # this does the pre-processing step

#         sim_train = kernal.transform(train_mod)
#         sim_test = kernal.transform(test_mod,train_mod)  # row = train, col = test

#         # Configure Kernel PCA for precomputed kernels
#         kpca = KernelPCA(kernel='precomputed')

#         # Transform data using Kernel PCA
#         X_kpca_train = kpca.fit_transform(sim_train)
#         X_kpca_test = kpca.transform(sim_test)

#     elif model == "mean":
#         # Mean imputation
#         imputer = SimpleImputer(strategy='mean')

#         # Fit the imputer on the training data and transform both training and test data
#         X_kpca_train = imputer.fit_transform(train_X)
#         X_kpca_test = imputer.transform(test_X)


#     elif model == "mice":
#         # MICE imputation
#         imputer = IterativeImputer()

#         # Fit the imputer on the training data and transform both training and test data
#         X_kpca_train = imputer.fit_transform(train_X)
#         X_kpca_test = imputer.transform(test_X)
#     elif model == "rbfn":

#         y_pred = rbfn(train_X,test_X,train_Y,test_Y)

#         # acc = accuracy_score(test_Y, y_pred)
#         # f1 = f1_score(test_Y, y_pred, average='macro')

#         exit()

#     elif model == "genrbf":
#         C = 1
#         gamma = 1.e-3
#         index_train = np.arange(train_X.shape[0])
#         index_test = np.arange(test_X.shape[0])

#         #X = np.concatenate((X_train, X_test), axis=0)
#         #del X_train, X_test

#         index_train = index_train.astype(np.intc)
#         index_test = index_test.astype(np.intc)
#         imputer = SimpleImputer(strategy='mean')
#         train_X_imputed = imputer.fit_transform(train_X)


#         m = np.mean(train_X_imputed, axis=0)
#         cov = np.cov(train_X_imputed, rowvar=False)

#         rbf_ker = rbf.RBFkernel(m, cov, train_X)


#         S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
#                                                                                         rbf_ker.S)

#         S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)

        

#         train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)

#         # test
#         S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
#         test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
#                                 completeDataId_test_new, completeDataId_train_new)


#         # Configure Kernel PCA for precomputed kernels
#         precomputed_svm = SVC(C=C, kernel='precomputed')
#         precomputed_svm.fit(train, train_Y)
#         y_pred = precomputed_svm.predict(test)
#         acc = accuracy_score(test_Y, y_pred)
#         f1 = f1_score(test_Y, y_pred, average='macro')

#     # Print mean F1 score and accuracy

#         print(f"Mean CV F1 Score {model}: {np.mean(f1):.4f}")
#         print(f"Mean CV Accuracy {model}: {np.mean(acc):.4f}")
        
#         return f1,acc
