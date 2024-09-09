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
from rbfn_model import run_rbfn
# from sklearn.svm import SVC
# from sklearn.decomposition import KernelPCA
# from mass import Modify_Kernel as MKernel
# import numpy as np
# from rbfn_model import rbfn



from sklearn.model_selection import StratifiedKFold

def run(dataset, missing_type, model, missing_rates, y):
    
    na_path = f"dataset_nan/{dataset}/{missing_type}/"

    if missing_rates is None: 
        missing_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    all_results = {}

    for rate in missing_rates:
        data_na = np.load(na_path + f"{rate}.npy")
        
        skf = StratifiedKFold(n_splits=5)
        results_list = []

        for id_acc, (trn_index, test_index) in enumerate(skf.split(data_na, y)):
            
            X_train, X_test = data_na[trn_index], data_na[test_index]
            y_train, y_test = y[trn_index], y[test_index]
            
            # Run the model and get the evaluation results
            results = run_model(model, X_train, X_test, y_train, y_test)
            results_list.append(results)
        
        # Aggregate results for this missing rate
        all_results[rate] = aggregate_results(results_list)

    return all_results

def run_model(model, X_train, X_test, y_train, y_test):
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
        train,test = run_genrbf(X_train, X_test)
        results = SVC_evaluation(train, y_train, test, y_test, kernel="precomputed")
    elif model == "rbfn":
        results = run_rbfn(X_train, X_test, y_train, y_test)

    return results

def SVC_evaluation(X_train, y_train, X_test, y_test, kernel="rbf"):
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