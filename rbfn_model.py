#!/usr/bin/env python
import os
from tqdm import tqdm
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
#tf.disable_v2_behavior()
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)



def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def scaler_range(X, feature_range=(-1, 1), min_x=None, max_x=None):
    if min_x is None:
        min_x = np.nanmin(X, axis=0)
        max_x = np.nanmax(X, axis=0)

    X_std = (X - min_x) / (max_x - min_x)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return X_scaled, min_x, max_x
# Create model




def neural_net(data, weights, means_, covs, s, x_, w_, gamma, n_d, n_h):
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

        new_data = tf.where(where_isnan, tf.reshape(tf.tile(means_[n_comp, :], [size[0]]), [-1, size[1]]), data)
        new_cov = tf.where(where_isnan, tf.reshape(tf.tile(covs_[n_comp, :], [size[0]]), [-1, size[1]]), tf.zeros([size[0], size[1]]))
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




# Parameters
#path_dir = FLAGS.data_dir
learning_rate = 0.001
batch_size = 32
training_epochs = 100

# Network Parameters
n_hidden_1 = 10
n_distribution = 3





def run_rbfn(train_x,test_x,train_y,test_y):

    n_features = train_x.shape[1]

    train_x, minx, maxx = scaler_range(train_x, feature_range=(-1, 1))

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    complate_data = imp.fit_transform(train_x)
    gmm = GaussianMixture(n_components=n_distribution, covariance_type='diag').fit(complate_data)

    mean_sel = complate_data[np.random.choice(complate_data.shape[0], size=n_hidden_1, replace=True), :]
    del complate_data, imp

    gmm_weights = np.log(gmm.weights_.reshape((-1, 1)))
    gmm_means = gmm.means_
    gmm_covariances = gmm.covariances_

    del gmm


    # Reshape labels if necessary
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Symbols (placeholders in TensorFlow 1.x style)
    z = tf.compat.v1.placeholder(shape=[None, n_features], dtype=tf.float32)
    y = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)

    # Initialize the variables with the appropriate initial values
    weights = tf.Variable(initial_value=gmm_weights, dtype=tf.float32)
    means = tf.Variable(initial_value=gmm_means, dtype=tf.float32)
    covs = tf.Variable(initial_value=gmm_covariances, dtype=tf.float32)

    x = tf.Variable(initial_value=mean_sel, dtype=tf.float32)
    s = tf.Variable(initial_value=tf.random.normal(shape=(n_hidden_1, n_features), mean=0., stddev=1.),
                    dtype=tf.float32)

    w = init_weights((n_hidden_1, 1))
    gamma = tf.Variable(initial_value=tf.random.normal(shape=(1,), mean=1., stddev=1.), dtype=tf.float32)

    try:
    # Construct the model
        predict = neural_net(z, weights, means, covs, s, x, w, gamma, n_distribution, n_hidden_1)
    except:
        results = {
            "accuracy": np.nan,
            "f1_score": np.nan,
        }

        return results


    # Mean squared error
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=y))

    # Gradient descent optimizer
    l_r = learning_rate
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(l_r).minimize(cost)

    # Initialize the variables
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        min_cost = np.inf
        n_cost_up = 0
        prev_train_cost = np.inf

        # Use tqdm to wrap the range of epochs
        for epoch in tqdm(range(training_epochs), desc="Training Epochs"):

            for batch_idx in range(0, train_y.shape[0], batch_size):
                x_batch = train_x[batch_idx:batch_idx + batch_size, :]
                y_batch = train_y[batch_idx:batch_idx + batch_size]

                sess.run(optimizer, feed_dict={z: x_batch, y: y_batch})

            curr_train_cost = sess.run(cost, feed_dict={z: train_x, y: train_y})
            if epoch > 10 and (prev_train_cost - curr_train_cost) < 1e-4 < l_r:
                l_r = l_r / 2.
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(l_r).minimize(cost)
            prev_train_cost = curr_train_cost

        # Evaluate the final model on the test set
        pred = tf.nn.sigmoid(predict)
        test_predictions = sess.run(tf.round(pred), feed_dict={z: test_x})

        # Convert predictions and true labels to numpy arrays
        y_pred = test_predictions.flatten()
        y_true = test_y.flatten()

        y_pred = np.nan_to_num(y_pred, nan=-1)
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        # Store the results in a dictionary
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
        }

        return results
