from ctypes import c_float

import numpy as np

try:
    import pymp
    pymp_found = True
except ImportError as e:
    pymp_found = False


from .equal_freq_discretization import EqualFrequencyDiscretizer


class IMPK_Kernel:
    def __init__(self, nbins = None, stats = None):
        self.nbins_ = nbins
        self.stats_ = stats
        self.bin_counts_ = None

    def build_model(self, train, test):

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

        self.ndata_ = len(train)
        self.ndim_ = len(train[0])

        if self.nbins_ is None:
            self.nbins_ = int(np.log2(self.ndata_) + 1)

        self.dimVec_ = np.array([i for i in range(self.ndim_)])
        self.discretiser_ = EqualFrequencyDiscretizer(train, self.nbins_, self.stats_)
        self.bin_cuts_, self.bin_counts_ = self.discretiser_.get_bin_cuts_counts()
        self.num_bins_ = self.discretiser_.get_num_bins()
        self.bin_dissimilarities_ = get_bin_dissimilarity()

        new_test = []

        for i in range(len(test)):
            new_test.append(self.discretiser_.get_bin_id(test[i, :]))

        return self.discretiser_.get_data_bin_id(), np.array(new_test, dtype = c_float, order = "C")

    def set_nbins(self, nbins):
        self.nbins_ = nbins

    def transform(self, train, test=None):

        def re_assign_bin(index, ref_bin):
            bin_count = self.bin_counts_[index]
            
            # Check if stats is valid and type is "Numeric" or "Ordinal"
            if (self.stats_ is None) or (self.stats_["attribute"][index]["type"] in ["Numeric", "Ordinal"]):

                # Bins less than or equal to ref_bin
                left_bins = bin_count[:int(ref_bin) + 1]
                # Bins greater than or equal to ref_bin
                right_bins = bin_count[int(ref_bin):]
                
                # Calculate the total number of data points on each side
                left_total = sum(left_bins)
                right_total = sum(right_bins)

                # Find which side has more data and return appropriate bin index
                if left_total > right_total:
                    return 0  # Assign to the leftmost bin (0 index)
                else:
                    return len(bin_count) - 1  # Assign to the rightmost bin
            else:
                # For Nominal types, assign to the bin with the most data
                max_bin = np.argmax(bin_count)
                return max_bin  # Return the index of the bin with the most data



        def convert(imput_x,index_x, imput_y,index_y):
            x_bin_ids = imput_x[index_x] # imput row
            y_bin_ids = imput_y[index_y] # imput row
            # Check if -1 exists in either x_bin_ids or y_bin_ids
            if -1 in x_bin_ids or -1 in y_bin_ids:
                for col, x_bin_id in enumerate(x_bin_ids):
                    if x_bin_id == -1 and y_bin_ids[col] == -1:
                        y_bin_ids[col] = self.nbins_
                        x_bin_ids[col] = 0
                    elif x_bin_id == -1: # if X_bin id is missing
                        x_bin_ids[col] = re_assign_bin(col, y_bin_ids[col])
                    elif y_bin_ids[col] == -1: # if X_bin id is missing
                        y_bin_ids[col]= re_assign_bin(col, x_bin_ids[col])
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

        pymp.config.nested = True

        if pymp_found:
            if test is None: # train similarity
                d = pymp.shared.array((len(train), len(train)))
                x_x = pymp.shared.array((len(train)))

                with pymp.Parallel() as p1:
                    for i in p1.range(len(train)):
                        temp_train_i, temp_train_j = convert(train,i, train,i)
                        x_x[i] = dissimilarity(temp_train_i, temp_train_j)

                with pymp.Parallel() as p1:
                    with pymp.Parallel() as p2:
                        for i in p1.range(len(train)):
                            for j in p2.range(i, len(train)):
                                temp_train_i, temp_train_j = convert(train,i, train,j)
                                x_y = dissimilarity(temp_train_i, temp_train_j)
                                d[i][j] = (2.0 * x_y) / (x_x[i] + x_x[j])
                                d[j][i] = d[i][j]
                                
            else: # test similarity similarity
                d = pymp.shared.array((len(train), len(test)))
                y_y = pymp.shared.array(len(test))

                # Parallel computation of y_y (self-dissimilarity of test set)
                with pymp.Parallel() as p1:
                    for i in p1.range(len(test)):
                        y_y[i] = dissimilarity(test[i], test[i])

                # Parallel computation of d matrix (train vs test dissimilarity)
                with pymp.Parallel() as p1:
                    for i in p1.range(len(train)):
                        # Convert train[i] only once, outside the inner loop
                        temp_train_i, temp_train_j = convert(train,i, train,i)
                        x_x = dissimilarity(temp_train_i, temp_train_j)

                        for j in range(len(test)):  # No need for additional parallelism here
                            # Convert only train[i] and test[j] for cross-dissimilarity
                            temp_train, temp_test = convert(train,i, test,j)
                            x_y = dissimilarity(temp_train, temp_test)

                            # Update the shared dissimilarity matrix
                            d[i][j] = (2.0 * x_y) / (x_x + y_y[j])

        else:
            if test is None:
                d = np.empty((len(train), len(train)))
                x_x = [0.0 for i in range(len(train))]

                for i in range(len(train)):
                    train[i], train[i] = convert(train,i, train,i) 
                    x_x[i] = dissimilarity(train[i], train[i])

                for i in range(len(train)):
                    for j in range(i, len(train)):
                        train[i], train[j] = convert(train,i, train,j) 
                        x_y = dissimilarity(train[i], train[j])

                        d[i][j] = (2.0 * x_y) / (x_x[i] + x_x[j])
                        d[j][i] = d[i][j]
            else:
                d = np.empty((len(train), len(test)))
                y_y = [0.0 for i in range(len(test))]
                for i in range(len(test)):
                    temp_test_i, temp_test_j = convert(test,i, test,i) 
                    y_y[i] = dissimilarity(temp_test_i, temp_test_j)

                for i in range(len(train)):
                    # Precompute dissimilarities for the train set using converted values
                    temp_train_i, temp_train_j = convert(train,i,train,i)
                    x_x = dissimilarity(temp_train_i, temp_train_j)

                    for j in range(len(test)):
                        # Convert train[i] and test[j] once for the cross-dissimilarity
                        temp_train, temp_test = convert(train,i, test,j)
                        
                        # Compute cross-dissimilarity
                        x_y = dissimilarity(temp_train, temp_test)

                        # Update d[i][j] using the precomputed self-dissimilarities
                        d[i][j] = (2.0 * x_y) / (x_x + y_y[j])

        return np.array(d)
