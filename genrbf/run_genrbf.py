
from .genRBF_source import RBFkernel as rbf
from .genRBF_source import cRBFkernel as fun
from sklearn.impute import SimpleImputer
import numpy as np

from sklearn.impute import SimpleImputer

def run_genrbf(data_na):
    gamma = 1.e-3

    # 1. Use mean imputer to impute the missing data -> X
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(data_na)

    # 2. Calculate the mean and covariance matrix
    m = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Prepare index for training
    index_train = np.arange(X.shape[0])
    index_train = index_train.astype(np.intc)

    # Train
    rbf_ker = rbf.RBFkernel(m, cov, X)
    S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_train, index_train,
                                                                                   rbf_ker.S)
    S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)

    train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)
    return train