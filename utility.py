param_value = None
data_stats = None
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from genrbf.run_genrbf import run_genrbf
# from sklearn.svm import SVC
# from sklearn.decomposition import KernelPCA
# from mass import Modify_Kernel as MKernel
# import numpy as np
# from genRBF_source import RBFkernel as rbf
# from genRBF_source import cRBFkernel as fun
# from ctypes import c_float
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from rbfn_model import rbfn



def run(dataset, missing_type, model, missing_rates, y):
    
    na_path = f"dataset_nan/{dataset}/{missing_type}/"

    if missing_rates is None: 
        missing_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    all_results = {}

    for rate in missing_rates:
        data_na = np.load(na_path + f"{rate}.npy")
        
        # Run the model and get the evaluation results
        results = run_model(model, data_na, y)

        # Store the results with the missing rate as the key
        all_results[rate] = results

    return all_results

def run_model(model, data_na, y):
    if model == "mean":
        imputer = SimpleImputer(strategy='mean')
        data_na = imputer.fit_transform(data_na)
        results = SVC_evaluation(data_na, y)

    elif model == "mice":
        imputer = IterativeImputer()
        data_na = imputer.fit_transform(data_na)
        results = SVC_evaluation(data_na, y)

    elif model == "genrbf":
        representation = run_genrbf(data_na)
        results = SVC_evaluation(representation, y,kernel="precomputed")
        

    return results

def SVC_evaluation(X, y, kernel="rbf"):
    C = 1

    # Define the SVC model
    precomputed_svm = SVC(C=C, kernel=kernel)

    # Define the cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Accuracy and F1 scorers
    accuracy_scorer = make_scorer(accuracy_score)
    f1_scorer = make_scorer(f1_score, average='macro')

    # Cross-validation for accuracy
    accuracies = cross_val_score(precomputed_svm, X, y, cv=kf, scoring=accuracy_scorer)
    avg_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()

    # Cross-validation for F1 score
    f1_scores = cross_val_score(precomputed_svm, X, y, cv=kf, scoring=f1_scorer)
    avg_f1_score = f1_scores.mean()
    std_f1_score = f1_scores.std()

    # Save the results in a dictionary
    results = {
        "avg_accuracy": avg_accuracy,
        "std_accuracy": std_accuracy,
        "avg_f1_score": avg_f1_score,
        "std_f1_score": std_f1_score
    }

    return results

def run_test(data,missing_rate,mtype = "mcar",model = "mass",data_stats = None):
    X = data["X"]
    Y = data["Y"]
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


    train_X = make_missing(train_X,rate = missing_rate,type = mtype)
    test_X = make_missing(test_X,rate = missing_rate,type = mtype)
    
    if model == "mass":
        kernal = MKernel(None, data_stats)
        kernal.set_nbins(param_value)


        train_mod, test_mod = kernal.build_model(train_X, test_X)  # this does the pre-processing step

        sim_train = kernal.transform(train_mod)
        sim_test = kernal.transform(test_mod,train_mod)  # row = train, col = test

        # Configure Kernel PCA for precomputed kernels
        kpca = KernelPCA(kernel='precomputed')

        # Transform data using Kernel PCA
        X_kpca_train = kpca.fit_transform(sim_train)
        X_kpca_test = kpca.transform(sim_test)

    elif model == "mean":
        # Mean imputation
        imputer = SimpleImputer(strategy='mean')

        # Fit the imputer on the training data and transform both training and test data
        X_kpca_train = imputer.fit_transform(train_X)
        X_kpca_test = imputer.transform(test_X)


    elif model == "mice":
        # MICE imputation
        imputer = IterativeImputer()

        # Fit the imputer on the training data and transform both training and test data
        X_kpca_train = imputer.fit_transform(train_X)
        X_kpca_test = imputer.transform(test_X)
    elif model == "rbfn":

        y_pred = rbfn(train_X,test_X,train_Y,test_Y)

        # acc = accuracy_score(test_Y, y_pred)
        # f1 = f1_score(test_Y, y_pred, average='macro')

        exit()

    elif model == "genrbf":
        C = 1
        gamma = 1.e-3
        index_train = np.arange(train_X.shape[0])
        index_test = np.arange(test_X.shape[0])

        #X = np.concatenate((X_train, X_test), axis=0)
        #del X_train, X_test

        index_train = index_train.astype(np.intc)
        index_test = index_test.astype(np.intc)
        imputer = SimpleImputer(strategy='mean')
        train_X_imputed = imputer.fit_transform(train_X)


        m = np.mean(train_X_imputed, axis=0)
        cov = np.cov(train_X_imputed, rowvar=False)

        rbf_ker = rbf.RBFkernel(m, cov, train_X)


        S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
                                                                                        rbf_ker.S)

        S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)

        

        train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)

        # test
        S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
        test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
                                completeDataId_test_new, completeDataId_train_new)


        # Configure Kernel PCA for precomputed kernels
        precomputed_svm = SVC(C=C, kernel='precomputed')
        precomputed_svm.fit(train, train_Y)
        y_pred = precomputed_svm.predict(test)
        acc = accuracy_score(test_Y, y_pred)
        f1 = f1_score(test_Y, y_pred, average='macro')

    # Print mean F1 score and accuracy

        print(f"Mean CV F1 Score {model}: {np.mean(f1):.4f}")
        print(f"Mean CV Accuracy {model}: {np.mean(acc):.4f}")
        
        return f1,acc


    #  Calculate F1 score and accuracy
    # Initialize the RandomForest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(rf, X_kpca_test, test_Y, cv=cv, scoring='f1_macro')
    acc_scores = cross_val_score(rf, X_kpca_test, test_Y, cv=cv, scoring='accuracy')

    # Print mean F1 score and accuracy

    print(f"Mean CV F1 Score {model}: {np.mean(f1_scores):.4f}")
    print(f"Mean CV Accuracy {model}: {np.mean(acc_scores):.4f}")
    return np.mean(f1_scores),np.mean(acc_scores)