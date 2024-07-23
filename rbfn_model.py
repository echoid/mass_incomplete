#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


# read data_rbfn potentially with missing values
def read_data(path, sep=',', val_type='f8'):
    return np.genfromtxt(path, dtype=val_type, delimiter=sep)


def scaler_range(X, feature_range=(-1, 1), min_x=None, max_x=None):
    if min_x is None:
        min_x = np.nanmin(X, axis=0)
        max_x = np.nanmax(X, axis=0)

    X_std = (X - min_x) / (max_x - min_x)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return X_scaled, min_x, max_x


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


# Create model
def neural_net(data, weights, means_, covs, s, x_, w_, gamma, n_d, n_h):
    # (z, weights, means, covs, s, x, w, gamma, n_distribution, n_hidden_1)

    """
    # Z = (?, nfeatures) 
    # weights, means, covs are parameters from GMM
    # s -> shape=(25, 19) number of nural
    # x -> initial for mean (25, 19)
    # gamma -> random, n_dis->predefine, n_neuraon -> 25

    
    """
    gamma_ = tf.abs(gamma)
    s_ = tf.abs(s)
    weights_ = tf.nn.softmax(weights, axis=0)
    covs_ = tf.abs(covs)

    where_isnan = tf.is_nan(data)
    
    where_isfinite = tf.is_finite(data)
    size = tf.shape(data)

    Q = []
    layer_1 = [[] for _ in range(n_d)]
    for n_comp in range(n_d):
        # loop over three components

        new_data = tf.where(where_isnan, tf.reshape(tf.tile(means_[n_comp, :], [size[0]]), [-1, size[1]]), data)
        new_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs_[n_comp, :], [size[0]]), [-1, size[1]]), tf.zeros([size[0], size[1]]))

        # minus the mean from each coponent
        norm = tf.subtract(new_data, means_[n_comp, :])
        norm = tf.square(norm)


        q = tf.where(where_isfinite,
                     tf.reshape(tf.tile(tf.add(gamma_, covs_[n_comp, :]), [size[0]]), [-1, size[1]]),
                     tf.ones_like(data))
        norm = tf.div(norm, q)

        norm = tf.reduce_sum(norm, axis=1)

        q = tf.log(q)
        q = tf.reduce_sum(q, axis=1)

        q = tf.add(q, norm)

        norm = tf.cast(tf.reduce_sum(tf.cast(where_isfinite, tf.int32), axis=1), tf.float32)
        norm = tf.multiply(norm, tf.log(2 * np.pi))

        q = tf.add(q, norm)
        q = tf.multiply(q, -0.5)

        Q = tf.concat((Q, q), axis=0)

        for i in range(n_h):
            h_sig = tf.add(s_[i, :], new_cov)

            norm = tf.subtract(new_data, x_[i, :])
            norm = tf.square(norm)
            norm = tf.div(norm, h_sig)
            norm = tf.reduce_sum(norm, axis=1)

            h_sig = tf.log(h_sig)
            det = tf.reduce_sum(h_sig, axis=1)
            det = tf.add(tf.multiply(tf.to_float(size[1]), tf.log(2 * np.pi)), det)

            norm = tf.add(norm, det)
            norm = tf.multiply(norm, -0.5)

            norm = tf.exp(norm)

            layer_1[n_comp] = tf.concat((layer_1[n_comp], norm), axis=0)

    Q = tf.reshape(Q, shape=(n_d, -1))
    Q = tf.add(Q, tf.log(weights_))
    Q = tf.nn.softmax(Q, axis=0)

    Q = tf.transpose(Q)
    Q = tf.tile(Q, [n_h, 1])

    layer_1 = tf.stack(layer_1, axis=1)
    layer_1 = tf.multiply(layer_1, Q)
    layer_1 = tf.reduce_sum(layer_1, axis=1)

    layer_1 = tf.reshape(layer_1, shape=(n_h, -1))

    layer_1 = tf.div(layer_1, tf.reduce_sum(layer_1, axis=0))
    out_layer_2 = tf.matmul(layer_1, w_, transpose_a=True)
    return out_layer_2
## main functions

def rbfn(train_X,test_X,train_Y,test_Y):
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--data_dir', type=str, default='./', help='Directory for input data_rbfn')

#    FLAGS, unparsed = parser.parse_known_args()

    # Parameters
    path_dir = "./data/bands/"
    learning_rate = 0.1
    batch_size = 50
    training_epochs = 2 #500

    # Network Parameters
    n_hidden_1 = 25
    n_distribution = 3

    data = read_data(os.path.join(path_dir, '_data.txt'))
    data, minx, maxx = scaler_range(data, feature_range=(-1, 1))

    labels = read_data(os.path.join(path_dir, '_labels.txt'))
    labels = np.where(labels == -1, 0, labels)

    print(train_X.shape,test_X.shape,train_Y.shape,test_Y.shape)
    print(data)
    print(labels)

    print(train_X)
    print(test_Y)



    n_features = data.shape[1]

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    complate_data = imp.fit_transform(data)
    gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(complate_data)
    

    mean_sel = complate_data[np.random.choice(complate_data.shape[0], size=n_hidden_1, replace=True), :]

    del complate_data, imp

    gmm_weights = np.log(gmm.weights_.reshape((-1, 1)))
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_

    del gmm


    X_train = data
    X_lab = labels
    train_index, valid_index = next(StratifiedKFold(n_splits=2).split(X_train, X_lab))

    train_x = X_train[train_index, :]
    valid_x = X_train[valid_index, :]
    test_x = data

    train_y = np.reshape(X_lab[train_index], (-1, 1))
    valid_y = np.reshape(X_lab[valid_index], (-1, 1))
    test_y = np.reshape(labels, (-1, 1))

    # Symbols
    z = tf.placeholder(shape=[None, n_features], dtype=tf.float32) # empty place holder shape=(?, number of features)
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32)


    weights = tf.Variable(initial_value=gmm_weights, dtype=tf.float32) # initial gmm weights
    means = tf.Variable(initial_value=gmm_means, dtype=tf.float32)
    covs = tf.Variable(initial_value=gmm_covariances, dtype=tf.float32)


    x = tf.Variable(initial_value=mean_sel, dtype=tf.float32) # initial the mean? what's for?
    s = tf.Variable(initial_value=tf.random_normal(shape=(n_hidden_1, n_features), mean=0., stddev=1.),
                    dtype=tf.float32) # shape=(25, 19) why?
    
    # random number initial the weight for w (25, 1)
    w = init_weights((n_hidden_1, 1))

    # just one single parameter gamma
    gamma = tf.Variable(initial_value=tf.random_normal(shape=(1,), mean=1., stddev=1.), dtype=tf.float32)

    # Construct model
    predict = neural_net(z, weights, means, covs, s, x, w, gamma, n_distribution, n_hidden_1)


    # Mean squared error
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=y))

    l_r = learning_rate
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    nr_epoch = 10

    val_x = None
    val_s = None
    val_w = None
    val_weights = None
    val_means = None
    val_covs = None
    val_gamma = None

    with tf.Session() as sess:
        sess.run(init)

        min_cost = np.inf
        n_cost_up = 0

        prev_train_cost = np.inf

        for epoch in range(training_epochs):
            print("Epoch nr {:d}".format(epoch))

            for batch_idx in range(0, train_y.shape[0], batch_size):
                x_batch = train_x[batch_idx:batch_idx + batch_size, :]
                y_batch = train_y[batch_idx:batch_idx + batch_size, :]

                sess.run(optimizer, feed_dict={z: x_batch, y: y_batch})

            curr_train_cost = sess.run(cost, feed_dict={z: train_x, y: train_y})
            # print("train_cost =", curr_train_cost)
            if epoch > nr_epoch and (prev_train_cost - curr_train_cost) < 1e-4 < l_r:
                l_r = l_r / 2.
                optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)
            prev_train_cost = curr_train_cost

            curr_cost = sess.run(cost, feed_dict={z: valid_x, y: valid_y})
            if min_cost > curr_cost:
                min_cost = curr_cost
                n_cost_up = 0

                val_x = x.eval()
                val_s = s.eval()
                val_w = w.eval()
                val_weights = weights.eval()
                val_means = means.eval()
                val_covs = covs.eval()
                val_gamma = gamma.eval()
            elif epoch > nr_epoch:
                n_cost_up = n_cost_up + 1

            if n_cost_up == 5 and 1e-4 < l_r:
                l_r = l_r / 2.
                optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)
            elif n_cost_up == 10:
                break

        weights = tf.convert_to_tensor(val_weights)
        means = tf.convert_to_tensor(val_means)
        covs = tf.convert_to_tensor(val_covs)

        x = tf.convert_to_tensor(val_x)
        s = tf.convert_to_tensor(val_s)

        w = tf.convert_to_tensor(val_w)
        gamma = tf.convert_to_tensor(val_gamma)

        predict = neural_net(z, weights, means, covs, s, x, w, gamma, n_distribution, n_hidden_1)

        pred_train = tf.nn.sigmoid(predict).eval({z: train_x})

        #print("before sig",predict.values())
        pred_test = tf.nn.sigmoid(predict).eval({z: test_x})
 
        sys.stdout.flush()

        return np.rint(pred_test)


