param_value = None
data_stats = None
from mass import Modify_Kernel as MKernel
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun
from ctypes import c_float
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

def make_missing(data, rate = 0.1, type = "mcar"):
    if not rate:
        return data
    missing_rate = rate  # 10% of the data will be missing
    # Calculate the number of elements to set as missing
    total_elements = data.size
    missing_elements = int(total_elements * missing_rate)

    # Create a random mask
    #np.random.seed(1)

    if type == "mcar":
        mask_indices = np.random.choice(total_elements, missing_elements, replace=False)

        # Convert flat indices to multi-dimensional indices
        multi_indices = np.unravel_index(mask_indices, data.shape)

        # Set selected elements to NaN
        data[multi_indices] = np.nan
    elif type == "mnar":
        for col in range(data.shape[1]):
            column_data = data[:, col]
            median_value = np.percentile(column_data, rate)
            #print(len(upper_quantile_indices))
            upper_quantile_indices = np.where(column_data > median_value)[0]
            missingnum = int(missing_elements/data.shape[1])-1

            if len(upper_quantile_indices) <= missingnum:
                missingnum = len(upper_quantile_indices)

            selected_indices = np.random.choice(upper_quantile_indices,missingnum , replace=False)
            data[selected_indices, col] = np.nan


    return data


def run_test(data,missing_rate,mtype = "mcar",model = "mass",data_stats = None,dataname = None):
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

        # Step 2: Compute column mean and covariance matrix
        if dataname == "genrbf":
            m = np.loadtxt("data/rbf/mu.txt", delimiter= ' ')  # Change delimiter if needed
            cov = np.loadtxt("data/rbf/cov.txt", delimiter=' ')  # Change delimiter if needed
        else:
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