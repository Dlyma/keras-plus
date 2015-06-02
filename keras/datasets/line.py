# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/5/30
# used for generate linear datasets
def generate_data(sz, T, diff_k):
    data = []
    for i in range(sz):
        k = 1
        if diff_k:
            k = np.random.random()
        data.append(np.multiply(np.array(range(T+1)).astype(float)/T, k) + np.random.random()/T)
    data = np.array(data).reshape(sz, T+1, 1)
    X = data[:, :T]
    Y = data[:, -1]
    return (X, Y)

def load_data(sz, T, path="line.pkl", diff_k = True):
    data = []
    if not os.path.exists(path):
        data = generate_data(sz, T, diff_k)
        cPickle.dump(data, open(path, 'wb'))
    else:
        data = cPickle.load(open(path, 'rb'))
        assert(data[0].shape[0] == sz)
        assert(data[0].shape[1] == T)

    return data #(X, Y)
