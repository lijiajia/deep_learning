#!/usr/bin/env python
# coding=utf-8


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def load_dataset():
    with gzip.open('/home/lijiajia/work/myproject/deep_learning/src/data/mnist.pkl.gz', 'rb') as f:
        data = pickle.load(f)

    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
#plt.imshow(X_train[0][0], cmap=cm.binary)
#plt.savefig('test.jpg')

net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv2d1', layers.Conv2DLayer),
        ('maxpool1', layers.MaxPool2DLayer),
        ('conv2d2', layers.Conv2DLayer),
        ('maxpool2', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('dense', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape = (None, 1, 28, 28),
    conv2d1_num_filters = 32,
    conv2d1_filter_size = (5, 5),
    conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
    conv2d1_W = lasagne.init.GlorotUniform(),
    maxpool1_pool_size = (2, 2),
    
    conv2d2_num_filters = 32,
    conv2d2_filter_size = (5, 5),
    conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
    maxpool2_pool_size = (2, 2),

    dropout1_p = 0.5,

    dense_num_units = 256,
    dense_nonlinearity = lasagne.nonlinearities.rectify,

    dropout2_p = 0.5,

    output_nonlinearity = lasagne.nonlinearities.softmax,
    output_num_units = 10,

    update = nesterov_momentum,
    update_learning_rate = 0.01,
    update_momentum = 0.9,
    max_epochs = 10,
    verbose = 1,
)

nn = net1.fit(X_train, y_train)

preds = net1.predict(X_test)
