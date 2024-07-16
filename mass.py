from ctypes import c_float
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from bisect import bisect_left
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun
from ctypes import c_float

import numpy as np

def max_distance(num, compare_to):
    # Calculate absolute differences
    distance_to_zero = abs(num - 0)
    distance_to_compare = abs(num - compare_to)

    # Determine which is closer
    if distance_to_zero > distance_to_compare:
        return 0
    elif distance_to_zero < distance_to_compare:
        return compare_to-1
    else:
        return num

# try:
#     import pymp
#     pymp_found = True
# except ImportError as e:
#     pymp_found = False


#from .equal_freq_discretization import EqualFrequencyDiscretizer


class Modify_Kernel:
    def __init__(self, nbins = None, stats = None):
        self.nbins_ = nbins
        self.stats_ = stats

    def build_model(self, train, test):

        # if data missing, 0 will be inputed
        def get_bin_dissimilarity():
            bin_dissim = [[] for i in range(self.ndim_)]
            max_num_bins = max(self.num_bins_)

            for i in range(self.ndim_):
                n_bins = self.num_bins_[i]
                bin_cf = [0 for j in range(n_bins)]
                cf = 0

                if (self.stats_ is not None) and ("Nominal" in self.stats_["attribute"][i]["type"]):
                    for j in range(n_bins):
                        bin_cf[j] = self.bin_counts_[i][j]
                else:
                    for j in range(n_bins):
                        cf = cf + self.bin_counts_[i][j]
                        bin_cf[j] = cf

                b_mass = [[0.0 for j in range(max_num_bins)] for k in range(max_num_bins)]

                for j in range(n_bins):
                    for k in range(j, n_bins):
                        if (self.stats_ is not None) and ("Nominal" in self.stats_["attribute"][i]["type"]):
                            if j == k:
                                prob_mass = (bin_cf[k] + 1) / (self.ndata_ + n_bins)
                            else:
                                prob_mass = (bin_cf[k] + bin_cf[j] + 1) / (self.ndata_ + n_bins)
                        else:
                            prob_mass = (bin_cf[k] - bin_cf[j] + self.bin_counts_[i][j] + 1) / (self.ndata_ + n_bins)

                        b_mass[j][k] = np.log(prob_mass)
                        b_mass[k][j] = b_mass[j][k]

                bin_dissim[i] = b_mass
            return np.array(bin_dissim)

        self.ndata_ = len(train) # number of train instance
        self.ndim_ = len(train[0]) # number of train column

        if self.nbins_ is None: # pre-define bin numbers
            self.nbins_ = int(np.log2(self.ndata_) + 1)


        self.dimVec_ = np.array([i for i in range(self.ndim_)])
        self.discretiser_ = EqualFrequencyDiscretizer(train, self.nbins_, self.stats_)
        self.bin_cuts_, self.bin_counts_ = self.discretiser_.get_bin_cuts_counts()
        self.num_bins_ = self.discretiser_.get_num_bins()
        self.bin_dissimilarities_ = get_bin_dissimilarity()

        new_test = []

        for i in range(len(test)):
            # make each column into bin_id
            new_test.append(self.discretiser_.get_bin_id(test[i, :]))
        
        
        return self.discretiser_.get_data_bin_id(), np.array(new_test, dtype = c_float, order = "C")

    def set_nbins(self, nbins):
        self.nbins_ = nbins

    def transform(self, train, test=None):
        def convert(x_bin_ids, y_bin_ids):
            if -1 in x_bin_ids or -1 in y_bin_ids:
                for i, bin_id in enumerate(x_bin_ids):
                    if bin_id == -1:
                        #print("Before",i,bin_id,y_bin_ids[i])
                        x_bin_ids[i] = max_distance(bin_id, self.nbins_)
                        #print("After",i,x_bin_ids[i],y_bin_ids[i])
                    elif y_bin_ids[i] == -1:
                        #print("Before",i,bin_id,y_bin_ids[i])
                        y_bin_ids[i] = max_distance(bin_id, self.nbins_)
                        #print("After",i,x_bin_ids[i],y_bin_ids[i])
                    elif (bin_id == -1) and (y_bin_ids[i] == -1):
                        #print("Before",i,bin_id,y_bin_ids[i])
                        y_bin_ids[i] = self.nbins_
                        x_bin_ids[i] = 0
                        #print("After",i,x_bin_ids[i],y_bin_ids[i])
                return x_bin_ids, y_bin_ids
            else:
                return x_bin_ids, y_bin_ids
        def dissimilarity(x_bin_ids, y_bin_ids):
            len_x, len_y = len(x_bin_ids), len(y_bin_ids)

            # check the vector size
            if (len_x != self.ndim_) or (len_y != self.ndim_):
                raise IndexError("Number of columns does not match.")

            m_dissim = self.bin_dissimilarities_[self.dimVec_, x_bin_ids.astype(int), y_bin_ids.astype(int)]
            return np.sum(m_dissim) / self.ndim_

        

        if test is None:
            d = np.empty((len(train), len(train)))
            x_x = [0.0 for i in range(len(train))]
            x_xi = [0.0 for i in range(len(train))]
            x_xj = [0.0 for i in range(len(train))]

            for i in range(len(train)):
                for j in range(i, len(train)):
                    train[i], train[j] = convert(train[i], train[j]) 
                    # updated i and j
                    x_y = dissimilarity(train[i], train[j])
                    x_xi[i] = dissimilarity(train[i], train[i])
                    x_xj[j] = dissimilarity(train[j], train[j])

                    d[i][j] = (2.0 * x_y) / (x_xi[i] + x_xj[j])
                    d[j][i] = d[i][j]
        else:
            d = np.empty((len(train), len(test)))
            y_y = [0.0 for i in range(len(test))]

            for i in range(len(train)):
                for j in range(len(test)):
                    train[i], test[j] = convert(train[i], test[j])
                    x_x = dissimilarity(train[i], train[i])
                    y_y[j] = dissimilarity(test[j], test[j])

                    x_y = dissimilarity(train[i], test[j])

                    d[i][j] = (2.0 * x_y) / (x_x + y_y[j])

        return np.array(d)


class EqualFrequencyDiscretizer(object):

  def __init__(self, data, nbins, stats):

    self.stats = stats
    self.n_data = len(data)
    self.n_dim = len(data[0])
    self.bin_cuts = [[] for i in range(self.n_dim)]
    self.bin_counts = [[] for i in range(self.n_dim)]
    self.data_bin_ids = np.array([[-1 for i in range(self.n_dim)] for i in range(self.n_data)])
    # initialized with -1
    self.num_bins = [0 for i in range(self.n_dim)]
    for i in range(self.n_dim):
      if (self.stats is None) or ("Numeric" in self.stats["attribute"][i]["type"]):
        column_data = data[:, i] # looking at each column
        # Filter out NaN values from the column
        temp = column_data[~np.isnan(column_data)]
        b_cuts, b_counts = self.equal_freq_histograms(temp, nbins)
        #b_cuts, b_counts = self.equal_freq_histograms(data[:, i], nbins)
        # b_cuts, b_counts = self.equal_freq_histograms_weka(data[:,i], nbins)
      else:
        b_cuts, b_counts = self.equal_freq_histograms_non_numeric(data[:, i], i)

      self.bin_cuts[i] = b_cuts
      self.bin_counts[i] = b_counts
      self.num_bins[i] = len(b_counts)
      for j in range(self.n_data):
        # for each column, look at each item
        if (self.stats is None) or ("Numeric" in self.stats["attribute"][i]["type"]):
            if np.isnan(data[j, i]):
              self.data_bin_ids[j,i] = -1
            else:
              self.data_bin_ids[j,i] = bisect_left(b_cuts[1:-1], data[j,i])
        else:
          self.data_bin_ids[j,i] = int(data[j,i])
  def get_bin_cuts_counts(self):
    return self.bin_cuts, self.bin_counts

  def get_num_bins(self):
    return self.num_bins 

  def get_data_bin_id(self):
    return np.array(self.data_bin_ids, dtype = c_float)

  def get_bin_id(self, x):
    x_bin_ids = [-1 for i in range(self.n_dim)]
    for i in range(self.n_dim):
      if (self.stats is None) or ("Numeric" in self.stats["attribute"][i]["type"]):
        if np.isnan(x[i]):
              x_bin_ids[i] = -1
        else:
          cuts = self.bin_cuts[i]
          x_bin_ids[i] = bisect_left(cuts[1:-1], x[i])
      else:
        x_bin_ids[i] = int(x[i])

    return np.array(x_bin_ids)

  def equal_freq_histograms_non_numeric(self, x, idx):
    # get unique values and counts
    unique_values, unique_value_counts = np.unique(x, return_counts = True)

    if (self.stats is not None) and ("Numeric" not in self.stats["attribute"][idx]["type"]):
      chk_cnt = []
      idx_chk = 0

      for i in range(len(self.stats["attribute"][idx]["values"])):
        if (idx_chk < len(unique_values)) and (unique_values[idx_chk] == i):
          chk_cnt.append(unique_value_counts[idx_chk])
          idx_chk += 1
        else:
          chk_cnt.append(0)

      unique_value_counts = chk_cnt

    # return the result
    return np.array([]), np.array(unique_value_counts)


  def equal_freq_histograms(self, x, nbins):

    b_cuts = []
    b_counts = []

    # get unique values and counts
    unique_values, unique_value_counts = np.unique(x, return_counts=True)
    num_unique_vals = len(unique_values)

    # start discretization
    x_size = len(x)
    exp_freq = x_size/nbins
    freq_count = 0
    last_freq_count = 0
    last_id = -1
    cut_point_id = 0

    b_cuts.append(unique_values[0] - (unique_values[1] - unique_values[0]) / 2)

    for i in range(num_unique_vals-1):
      freq_count += unique_value_counts[i]
      x_size -= unique_value_counts[i]
      # check if ideal bin count is reached
      if (freq_count >= exp_freq):
        # check if this one is worst than the last one
        if (((exp_freq - last_freq_count) < (freq_count - exp_freq)) and (last_id != -1) ):
          cut_point = (unique_values[last_id] + unique_values[last_id+1])/2
          # check if it worths merging the about to create bin with the last bin
          if (len(b_counts) > 1):
            if ((abs(b_counts[-1] + last_freq_count) - exp_freq) < abs(last_freq_count - exp_freq)):
              b_counts[-1] += last_freq_count
              b_cuts[-1] = cut_point
            else: 
              b_cuts.append(cut_point)
              b_counts.append(last_freq_count)
          else:
              b_cuts.append(cut_point)
              b_counts.append(last_freq_count)              
          freq_count -= last_freq_count
          last_freq_count = freq_count
          last_id = i
        else:
          b_cuts.append((unique_values[i] + unique_values[i+1])/2)
          b_counts.append(freq_count)
          freq_count = 0
          last_freq_count = 0
          last_id = -1
        # increase the counter
        cut_point_id += 1
        # exp_freq = (x_size + freq_count) / (nbins - cut_point_id)
      else:  
        last_id = i
        last_freq_count = freq_count

    # what to do with the last unique value frequency
    last_unique_value_count = unique_value_counts[i+1] 
    freq_count = freq_count + last_unique_value_count
    x_size -= unique_value_counts[i+1]

    # Just make sure that it is the last unique value
    if (x_size != 0):
      print('ERROR: Something is wrong, x_size should be 0 but x_size=%s' % (x_size))
      exit()
     
    # check if the next partition is required
    if ((last_id != -1) and (abs(exp_freq - last_unique_value_count) < abs(freq_count - exp_freq))):
      b_cuts.append((unique_values[last_id] + unique_values[last_id+1])/2)
      b_counts.append(last_freq_count)
      freq_count -= last_freq_count

    b_counts.append(freq_count)
     
    # check if the last partition can be merged with the one before
    if (len(b_counts) >= 2):    
      if (abs((b_counts[-2] + b_counts[-1]) - exp_freq) < abs(exp_freq - b_counts[-1])): 
         b_counts[-2] += b_counts[-1]
         del b_cuts[-1]
         del b_counts[-1]

    # check if it is worth merging the second last bin with the third last
    if (len(b_counts) >= 3):
      if (abs((b_counts[-3] + b_counts[-2]) - exp_freq) < abs(exp_freq - b_counts[-2])):
        b_counts[-3] += b_counts[-2]
        b_counts[-2] = b_counts[-1]
        del b_cuts[-2]
        del b_counts[-1]

    b_cuts.append(unique_values[num_unique_vals-1] + (unique_values[num_unique_vals-1] - unique_values[num_unique_vals-2]) / 2) 

    assert sum(b_counts) == len(x)
    assert len(b_cuts) == (len(b_counts) + 1)

    # return the result
    return np.array(b_cuts), np.array(b_counts)

  def equal_freq_histograms_weka(self, x, nbins): # WEKA Implementation
    b_cuts = []
    b_counts = []
    # get unique values and counts
    unique_values, unique_value_counts = np.unique(x, return_counts=True)
    num_unique_vals = len(unique_values)

    x_size = len(x)
    exp_freq = x_size/nbins
    freq_count = 0
    last_freq_count = 0
    last_id = -1
    cut_point_id = 0
    for i in range(num_unique_vals-1):
      freq_count += unique_value_counts[i]
      x_size -= unique_value_counts[i]
      # check if ideal bin count is reached
      if (freq_count >= exp_freq):
        # check if this one is worst than the last one
        if (((exp_freq - last_freq_count) < (freq_count - exp_freq)) and (last_id != -1) ):
          b_cuts.append((unique_values[last_id] + unique_values[last_id+1])/2)
          b_counts.append(last_freq_count)
          freq_count -= last_freq_count
          last_freq_count = freq_count
          last_id = i
        else:
          b_cuts.append((unique_values[i] + unique_values[i+1])/2)
          b_counts.append(freq_count)
          freq_count = 0
          last_freq_count = 0
          last_id = -1

        cut_point_id += 1
        exp_freq = (x_size + freq_count) / (nbins - cut_point_id)

      else:  
        last_id = i;
        last_freq_count = freq_count;

    freq_count += unique_value_counts[i+1]

    # what to do with the last unique value
    if ((cut_point_id < nbins) and (freq_count > exp_freq) and ((exp_freq - last_freq_count) < (freq_count - exp_freq))):
      b_cuts.append((unique_values[last_id] + unique_values[last_id+1])/2)
      b_counts.append(last_freq_count)
      b_counts.append(freq_count-last_freq_count)
    else:  
      b_counts.append(freq_count)

    return np.array(b_cuts), np.array(b_counts)
  


def make_missing(data, rate = 0.1, type = "mcar"):
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
            median_value = np.percentile(column_data, 60)
            #print(len(upper_quantile_indices))
            upper_quantile_indices = np.where(column_data > median_value)[0]
            missingnum = int(missing_elements/data.shape[1])

            selected_indices = np.random.choice(upper_quantile_indices,missingnum , replace=False)
            data[selected_indices, col] = np.nan


    return data
param_value = None
data_stats = None
from mass import Modify_Kernel as MKernel
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
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

    elif model == "genRBF":
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