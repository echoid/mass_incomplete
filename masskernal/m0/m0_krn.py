from ctypes import c_float

import numpy as np

try:
    import pymp
    pymp_found = True
except ImportError as e:
    pymp_found = False


from .equal_freq_discretization import EqualFrequencyDiscretizer


class M0_Kernel:
    def __init__(self, nbins = None, stats = None):
        self.nbins_ = nbins
        self.stats_ = stats

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
        def dissimilarity(x_bin_ids, y_bin_ids):
            len_x, len_y = len(x_bin_ids), len(y_bin_ids)

            # check the vector size
            if (len_x != self.ndim_) or (len_y != self.ndim_):
                raise IndexError("Number of columns does not match.")

            m_dissim = self.bin_dissimilarities_[self.dimVec_, x_bin_ids.astype(int), y_bin_ids.astype(int)]
            return np.sum(m_dissim) / self.ndim_

        pymp.config.nested = True

        if pymp_found:
            if test is None:
                d = pymp.shared.array((len(train), len(train)))
                x_x = pymp.shared.array((len(train)))

                with pymp.Parallel() as p1:
                    for i in p1.range(len(train)):
                        x_x[i] = dissimilarity(train[i], train[i])

                with pymp.Parallel() as p1:
                    with pymp.Parallel() as p2:
                        for i in p1.range(len(train)):
                            for j in p2.range(i, len(train)):
                                x_y = dissimilarity(train[i], train[j])

                                d[i][j] = (2.0 * x_y) / (x_x[i] + x_x[j])
                                d[j][i] = d[i][j]
            else:
                d = pymp.shared.array((len(train), len(test)))
                y_y = pymp.shared.array(len(test))

                with pymp.Parallel() as p1:
                    for i in p1.range(len(test)):
                        y_y[i] = dissimilarity(test[i], test[i])

                with pymp.Parallel() as p1:
                    with pymp.Parallel() as p2:
                        for i in p1.range(len(train)):
                            x_x = dissimilarity(train[i], train[i])

                            for j in p2.range(len(test)):
                                x_y = dissimilarity(train[i], test[j])

                                d[i][j] = (2.0 * x_y) / (x_x + y_y[j])
        else:
            if test is None:
                d = np.empty((len(train), len(train)))
                x_x = [0.0 for i in range(len(train))]

                for i in range(len(train)):
                    x_x[i] = dissimilarity(train[i], train[i])

                for i in range(len(train)):
                    for j in range(i, len(train)):
                        x_y = dissimilarity(train[i], train[j])

                        d[i][j] = (2.0 * x_y) / (x_x[i] + x_x[j])
                        d[j][i] = d[i][j]
            else:
                d = np.empty((len(train), len(test)))
                y_y = [0.0 for i in range(len(test))]

                for i in range(len(test)):
                    y_y[i] = dissimilarity(test[i], test[i])

                for i in range(len(train)):
                    x_x = dissimilarity(train[i], train[i])

                    for j in range(len(test)):
                        x_y = dissimilarity(train[i], test[j])

                        d[i][j] = (2.0 * x_y) / (x_x + y_y[j])

        return np.array(d)
