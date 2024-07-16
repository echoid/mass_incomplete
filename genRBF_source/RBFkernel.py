from copy import deepcopy

from genRBF_source import cRBFkernel as f
import numpy as np

__author__ = "Łukasz Struski"


# read data potentially with missing values
def read_data(path, sep=','):
    return np.genfromtxt(path, delimiter=sep)


def whiten_matrix(covariance_matrix):
    EPS = 1e-10
    # eigenvalue decomposition of the covariance matrix
    d, E = np.linalg.eigh(covariance_matrix)
    d = np.divide(1., np.sqrt(d + EPS))
    W = np.einsum('ij, j, kj -> ik', E, d, E)
    return W

def missingINFO(data, fill):
    S = []
    J = []
    JJ_ = set()
    completeDataId = []

    for i in range(data.shape[0]):
        x = []
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                x.append(j)
        size_ = len(x)
        if size_:
            for j in range(size_):
                k = x[j]
                data[i, k] = fill[k]
            if J:
                b1 = False
                for j in range(len(J)):
                    if size_ == len(J[j]):
                        b2 = True
                        for k in range(size_):
                            if x[k] != J[j][k]:
                                b2 = False
                                break
                        if b2:
                            b1 = True
                            break
                if b1:
                    S[j].append(i)
                else:
                    J.append(x)
                    S.append([i])
                    JJ_.update(x)
            else:
                J.append(x)
                S.append([i])
                JJ_.update(x)
        else:
            completeDataId.append(i)

    JJ = sorted(JJ_)
    return S, JJ, J, completeDataId

def trainTestID_1(test_id, train_id, S):
    size_ = len(S)
    S_test = [[] for _ in range(size_)]
    S_train = [[] for _ in range(size_)]
    completeDataId_test = []
    completeDataId_train = []

    size1 = len(test_id)
    size2 = len(train_id)
    size3 = max(size1, size2)

    for i in range(size3):
        check1 = i < size1
        check2 = i < size2

        for j in range(size_):
            if check1 and test_id[i] in S[j]:
                S_test[j].append(test_id[i])
                check1 = False
            elif check2 and train_id[i] in S[j]:
                S_train[j].append(train_id[i])
                check2 = False
            if not (check1 or check2):
                break

        if check1:
            completeDataId_test.append(test_id[i])
        if check2:
            completeDataId_train.append(train_id[i])

    return (S_train, S_test, completeDataId_train, completeDataId_test)

def updateSamples(train_id, S_train, completeDataId_train):
    size_ = len(S_train)
    S_train_new = [[] for _ in range(size_)]
    completeDataId_train_new = []

    my_dict = {train_id[i]: i for i in range(len(train_id))}

    S_train_new = [[my_dict[item] for item in sublist] for sublist in S_train]
    completeDataId_train_new = [my_dict[item] for item in completeDataId_train]

    return S_train_new, completeDataId_train_new

def p_trainTestID(test_id, S, completeDataId):
    size_ = len(S)
    size1 = 0
    S_test = [[] for _ in range(size_)]
    S_train = []
    completeDataId_test = []
    completeDataId_train = []

    # Compute S_test and completeDataId_test
    for item in test_id:
        check = True
        for j in range(size_):
            for k in range(len(S[j])):
                if item == S[j][k]:
                    S_test[j].append(item)
                    check = False
                    break
            if not check:
                break
        if check:
            completeDataId_test.append(item)

    # Compute S_train
    for i in range(size_):
        S_train.append([])
        l = 0
        size1 = len(S[i])
        for item in S_test[i]:
            for k in range(l, size1):
                if S[i][k] < item:
                    S_train[i].append(S[i][k])
                else:
                    l = k + 1
                    break
        for j in range(l, size1):
            S_train[i].append(S[i][j])

    # Compute completeDataId_train
    size1 = len(completeDataId)
    l = 0
    for item in completeDataId_test:
        for j in range(l, size1):
            if completeDataId[j] < item:
                completeDataId_train.append(completeDataId[j])
            else:
                l = j + 1
                break
    for i in range(l, size1):
        completeDataId_train.append(completeDataId[i])

    return S_train, S_test, completeDataId_train, completeDataId_test

import numpy as np
"""
Initialization: 
    The function initializes an empty matrix gramRBF of shape 
    (n_samples, n_samples)to store the computed Gram matrix.

Case I (diagonal): 
    It sets the diagonal elements of the Gram matrix to 1 and 
    the off-diagonal elements to 0.

Case II (no missing): 
    For pairs of samples with no missing values (completeDataId),
      it computes the pairwise similarity using the RBF kernel function
        and stores the result in the Gram matrix.

Case III (one missing): 
    For pairs of samples where one sample has missing values 
    (completeDataId[i]), it calculates the similarity based on the 
    available features and uses a weighted sum to handle missing values.

Case IV (two missing): 
    For pairs of samples where both samples have missing values (S[i], S[j]), 
    it computes the similarity based on the available features and uses a weighted
      sum to handle missing values.

Return: 
    Finally, the function returns the computed Gram matrix.

"""
def p_krenel_train(gamma, X, new_X, Z, G, S, J, JJ, completeDataId, Ps):
    n_samples, n_features = X.shape
    gramRBF = np.empty((n_samples, n_samples), dtype=float)

    # case I (diagonal)
    for i in range(n_samples):
        for j in range(n_samples):
            gramRBF[i, j] = 1. if i == j else 0.

    # case II (no missing)
    size_ = len(completeDataId)
    for i in range(size_):
        kk = completeDataId[i]
        for j in range(i + 1, size_):
            jj = completeDataId[j]
            scalar = 0.
            for ii in range(n_features):
                scalar += (X[kk, ii] - X[jj, ii]) * (Z[kk, ii] - Z[jj, ii])
            gramRBF[kk, jj] = np.exp(-gamma * scalar) #genRBF kernal
            gramRBF[jj, kk] = gramRBF[kk, jj]


    # case III (one missing)
    for i in range(size_):
        for id_ in range(len(S)):
            Gs = G[np.ix_(J[id_], J[id_])]
            z = len(J[id_])
            z = np.power(1 + 4 * gamma, z / 4.0) / np.power(1 + 2 * gamma, z / 2.0)

            ps = Ps[id_]

            temp_x = new_X[completeDataId[i], :] - new_X[S[id_], :]
            p = np.einsum('ij,kj->ik', ps, temp_x)
            r = np.einsum('ji,jk,ki->i', p, Gs, p)
            temp_x = X[completeDataId[i], :] - X[S[id_], :]
            temp_z = Z[completeDataId[i], :] - Z[S[id_], :]
            w = np.einsum('ij,ij->i', temp_x, temp_z) - (2 * gamma) / (1 + 2 * gamma) * r
            gramRBF[completeDataId[i], S[id_]] = z * np.exp(-gamma * w) #genRBF kernal
            gramRBF[S[id_], completeDataId[i]] = gramRBF[completeDataId[i], S[id_]]


    # case IV (two missing)
    size_ = len(S)
    for i in range(size_):
        Gs = G[np.ix_(J[i], J[i])]
        z = 1

        ps = Ps[i]

        for j in range(i, size_):
            if i == j:
                size2 = len(S[i])
                for ii in range(size2):
                    l = S[i][ii]
                    for jj in range(ii + 1, size2):
                        ll = S[i][jj]
                        temp_x = new_X[l, :] - new_X[ll, :]
                        p = np.einsum('ij,j->i', ps, temp_x)
                        r = np.einsum('i,ij,j', p, Gs, p)
                        scalar = 0.
                        for kk in range(n_features):
                            scalar += (X[l, kk] - X[ll, kk]) * (Z[l, kk] - Z[ll, kk])
                        w = scalar - 4 * gamma * r / (1 + 4 * gamma)
                        gramRBF[l, ll] = z * np.exp(-gamma * w) #genRBF kernal
                        gramRBF[ll, l] = gramRBF[l, ll]
            else:
                ps_j = Ps[j]

                J_ij = set(J[i])
                J_ij.update(J[j])
                J_ij = sorted(J_ij)
                size2 = len(JJ)
                Q = np.zeros((size2, size2))
                Q[J[i], :] += ps
                Q[J[j], :] += ps_j
                I = np.identity(size2)
                Q = I[np.ix_(J_ij, J_ij)] + 2 * gamma * Q[np.ix_(J_ij, J_ij)]
                Q = np.linalg.inv(Q)
                R = Q - I[np.ix_(J_ij, J_ij)]
                R = np.einsum('ij,jk->ik', G[np.ix_(J_ij, J_ij)], R)
                z = np.power(1 + 4 * gamma, (len(J[i]) + len(J[j])) / 4.) * np.sqrt(np.linalg.det(Q))

                ps_ij = np.linalg.inv(G[np.ix_(J_ij, J_ij)])
                ps_ij = np.einsum('ij,jk->ik', ps_ij, G[np.ix_(J_ij, JJ)])

                for ii in range(len(S[i])):
                    temp_x = new_X[S[i][ii], :] - new_X[S[j], :]
                    v = np.einsum('ij,kj->ki', ps_ij, temp_x)
                    temp_x = X[S[i][ii], :] - X[S[j], :]
                    temp_z = Z[S[i][ii], :] - Z[S[j], :]
                    w = np.einsum('ij,ij->i', temp_x, temp_z) - np.einsum('ij,jk,ik->i', v, R, v)
                    gramRBF[S[i][ii], S[j]] = z * np.exp(-gamma * w) #genRBF kernal
                    gramRBF[S[j], S[i][ii]] = gramRBF[S[i][ii], S[j]]
    return gramRBF


class RBFkernel(object):
    """
    Fast rbf kernel for missing data
    """

    def __init__(self, mean, covariance_matrix, data):
        np.atleast_2d(data)
        self.n_samples, self.n_features = data.shape
        self.data = data  # np.copy(data)
        self.mean = mean
        # self._info(self.mean)
        self._info(np.zeros(self.n_features))

        G = np.linalg.inv(covariance_matrix)
        reference = False
        if len(self.JJ) < self.n_features:
            self.G = G[np.ix_(self.JJ, self.JJ)]
            G_JJ_inv = np.linalg.inv(self.G)
            P_JJ = np.einsum('ij,jk->ik', G_JJ_inv, G[self.JJ, :])
            self.new_data = np.einsum('ij,kj->ki', P_JJ, self.data)
            self.mean = np.einsum('ij,j->i', P_JJ, self.mean)
            del G_JJ_inv, P_JJ

            self.Jx = deepcopy(self.J)
            f.updateFeatures(self.J, self.JJ)
        else:
            reference = True
            self.Jx = self.J
            self.G = np.copy(G)
            self.new_data = self.data
        self.Ps = f.change_data(self.data, self.new_data, self.mean, self.JJ, self.S, self.J, self.Jx, self.G,
                                reference=reference)
        self.Z = np.einsum('ij, kj->ki', G, self.data)
        del G

    def _info(self, fill):
        """
        This function collects informations about missing values and puts
        into missing coordinates from fill vector.
        :param data: dataset, numpy asrray, shape like (n_samples, n_features)
        :param fill: vector , numpy array, shape like (n_features)
        :return: list of indexes, list of missing sets, list of indexes of all points which have missing components
        """
        self.S, self.JJ, self.J, self.completeDataId = f.missingINFO(self.data, fill)
        #self.S, self.JJ, self.J, self.completeDataId = missingINFO(self.data, fill)
    def trainTestID(self, test_id):
        return f.trainTestID(test_id, self.S, self.completeDataId)
        #return p_trainTestID(test_id, self.S, self.completeDataId)




    def kernelTrain(self, gamma, train_id, S_train_new, completeDataId_train_new):
        return f.krenel_train(gamma, self.data[train_id, :], self.new_data[train_id, :], self.Z[train_id, :], self.G,
                              S_train_new, self.J, self.JJ, completeDataId_train_new, self.Ps)
        # return p_krenel_train(gamma, self.data[train_id, :], self.new_data[train_id, :], self.Z[train_id, :], self.G,
        #                       S_train_new, self.J, self.JJ, completeDataId_train_new, self.Ps)


    def kernelTest(self, gamma, test_id, train_id, S_test_new, S_train_new, completeDataId_test_new,
                   completeDataId_train_new):
        return f.krenel_test(test_id, train_id, gamma, self.data, self.new_data, self.Z, self.G, S_train_new,
                             S_test_new, self.J, self.JJ, completeDataId_train_new, completeDataId_test_new,
                             self.Ps)
        # return p_krenel_test(test_id, train_id, gamma, self.data, self.new_data, self.Z, self.G, S_train_new,
        #                      S_test_new, self.J, self.JJ, completeDataId_train_new, completeDataId_test_new,
        #                      self.Ps)
