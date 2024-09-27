from mpk import M0_Kernel
import numpy as np
from impk import IMPK_Kernel




def run_mpk(train_X, test_X, data_stats):

    # set the number of bins
    param_value = None  # use default: log2(num of inst) + 1
    m0_krn = M0_Kernel(param_value, data_stats)
    m0_krn.set_nbins(param_value)
    train, test = m0_krn.build_model(train_X, test_X)  # this does the pre-processing step
    # print(train.shape)
    # print(np.max(train, axis=0))
    # print(np.max(train, axis=1))
    print("- Sim: Train")   
    sim_train = m0_krn.transform(train)

    print("- Sim: Train/Test")
    sim_test = m0_krn.transform(train,test)

    return sim_train, sim_test.T


def run_impk(train_X, test_X, data_stats):
    pass