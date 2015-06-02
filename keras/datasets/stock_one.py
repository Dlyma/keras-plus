# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/5/22
# used for generate datasets for the stock
def generate_data(pkl_path, norm, sz, maxlen, mm, reverse):
    X = cPickle.load(open(pkl_path, 'rb'))[:mm+1]
    if reverse:
        X.reverse()
    X = np.array(X)
    if norm == 'minmax':
        mins = X.min(axis=0, keepdims=True)
        maxs = X.max(axis=0, keepdims=True) + 1e-4
        X = (X-mins) / (maxs - mins)
        X[X>1] = 0
        X = X*2 -1
    
    if maxlen is None:
        maxlen = X.shape[0]-1
    maxlen = min(maxlen+1, X.shape[0])
    assert(maxlen > 1)

    step = X.shape[0]/maxlen
    # data = np.zeros([step, maxlen, X.shape[1]])
    data = []
    for i in range(step):
        data.append( X[i:X.shape[0]:step][:maxlen])
    if sz is None:
        sz = len(data)
    while len(data) < sz:
        data.extend(data)
    data = np.array(data[:sz])
    data += np.random.random(data.shape)/100
    np.random.shuffle(data)

    X = data[:, :maxlen-1, :]
    Y = data[:, -1, :]
    return (X, Y) # [timerange/maxlen, maxlen, 5]

def load_data(pkl_path, norm='minmax', sz=None, maxlen=None, mm=5000, reverse=False):
    data = generate_data(pkl_path, norm, sz, maxlen, mm, reverse)
    return data
