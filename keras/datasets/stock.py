# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/5/22
# used for generate datasets for the adding problem
def generate_data(pkl_path, T, norm):
    dataset = []
    for f in os.listdir(pkl_path):
        data = cPickle.load(open(os.path.join(pkl_path, f), 'rb'))
        if len(data) > T:
            data = data[:T+1]
            data.reverse()
            dataset.append(data)
    
    dataset = np.array(dataset)
    if norm == 'minmax':
        mins = dataset.min(axis=1, keepdims=True)
        maxs = dataset.max(axis=1, keepdims=True) + 1e-4
        dataset = (dataset-mins.repeat(T+1, 1)) / np.repeat(maxs - mins, T+1, 1)
        dataset[dataset>1] = 0
    
    # add noise [0, 0.01]
    dataset += np.random.random(dataset.shape)/100

    X = dataset[:, :T, :]
    Y = dataset[:, -1, :]

    return (X, Y)

def load_data(pkl_path, T, path='stock.pkl', norm='minmax'):
    data = []
    if not os.path.exists(path):
        print(path, 'not exists', T)
        data = generate_data(pkl_path, T, norm)
        cPickle.dump(data, open(path, 'wb'))
    else:
        print(path, 'exists', T)
        data = cPickle.load(open(path, 'rb'))
        assert(data[0].shape[1] == T)

    return data #(X, Y)
