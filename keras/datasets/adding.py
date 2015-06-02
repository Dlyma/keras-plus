# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/5/22
# used for generate datasets for the adding problem
def generate_data(sz, T):
    NUM_SUM = 2
    X = np.zeros((sz*2, T, 2))
    X[:,:,0] = np.random.uniform(0, 1, [sz*2, T])
    Y = np.zeros(sz*2)
    for i in range(sz*2):
        mask = np.random.choice(range(T), NUM_SUM, replace=False)
        for j in mask:
            X[i, j, 1] = 1
            Y[i] = Y[i] + X[i, j, 0]
    X_train = X[:sz]
    y_train = Y[:sz]
    X_test = X[sz:]
    y_test = Y[sz:]
    return ((X_train, y_train), (X_test, y_test))

def load_data(T, sz, path="adding.pkl"):
    data = []
    if not os.path.exists(path):
        data = generate_data(sz, T)
        cPickle.dump(data, open(path, 'wb'))
    else:
        data = cPickle.load(open(path, 'rb'))
        assert(data[0][0].shape[0] == sz)
        assert(data[0][0].shape[1] == T)

    return data #(X_train, y_train), (X_test, y_test)
