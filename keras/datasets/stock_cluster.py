# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/5/22
# used for generate datasets for the stock
def generate_data(csv_path, norm, sz, maxlen, step, reverse, mean):
    X = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            line.remove('')
            X.append( [float(s) for s in line])
    
    if reverse:
        X.reverse()
    X = np.array(np.matrix(np.array(X)).transpose()) #nb_samples, T
    
    if maxlen is None:
        maxlen = (X.shape[1]-1)
    maxlen = min(maxlen+1, X.shape[1])/step
    assert(maxlen > 1)

    data = []
    X = X.reshape(X.shape[0], X.shape[1], 1)
    for i in range(step):
        data.extend( X[:, i:X.shape[1]:step][:, :maxlen])
    data = np.array(data[:sz])
    if mean:
        data = data.mean(axis=0, keepdims=True)
    if sz is None:
        sz = len(data)
    while len(data) < sz:
        data = data.repeat(2, axis=0)
    np.random.shuffle(data)
    data = data[:sz]
     
    mins = []
    maxs = []
    if norm == 'minmax':
        mins = data.min(axis=1, keepdims=True)
        maxs = data.max(axis=1, keepdims=True) + 1e-4
        data = (data-mins.repeat(data.shape[1], 1))/ np.repeat(maxs - mins, data.shape[1], 1)
        data[data>1] = 0
        data = data*2 -1
    
    data += np.random.random(data.shape)/100

    X = data[:, :maxlen-1, :]
    Y = data[:, -1, :]
    if norm == 'minmax':
        return (X, Y, mins, maxs)
    else:
        return (X, Y) # [timerange/maxlen, maxlen, 5]

def load_data(csv_path, norm='minmax', sz=None, maxlen=None, step=1, reverse=False, mean=False):
    data = generate_data(csv_path, norm, sz, maxlen, step, reverse, mean)
    return data
