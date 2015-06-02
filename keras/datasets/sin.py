# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/5/30
# used for generate linear datasets
def generate_data(sz, T, diff_start, diff_T):
    data = []
    for i in range(sz):
        start = 0
        if diff_start:
            start = np.random.random()
        Time = 2*np.pi
        times_of_loops = 5
        if diff_T:
            times_of_loops = np.random.randint(10) + 1
        data.append(np.sin((np.array(range(T+1)).astype(float)/T + start)*Time*times_of_loops) + np.random.random()/T)
    data = np.array(data).reshape(sz, T+1, 1)
    X = data[:, :T]
    Y = data[:, -1]
    return (X, Y)

def load_data(sz, T, path="sin.pkl", diff_start = False, diff_T = False):
    data = []
    if not os.path.exists(path):
        print(path, 'not exists')
        data = generate_data(sz, T, diff_start, diff_T)
        cPickle.dump(data, open(path, 'wb'))
    else:
        print(path, 'exists')
        data = cPickle.load(open(path, 'rb'))
        assert(data[0].shape[0] == sz)
        assert(data[0].shape[1] == T)

    return data #(X, Y)
