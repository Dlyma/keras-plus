# -*- coding: utf-8 -*-
import cPickle
import sys, os
import numpy as np

# written by zhaowuxia @ 2015/6/8
# used for generate datasets for the tianchi
def generate_data(csv_path, norm, sz, maxlen, step, reverse):
    X = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            date = line[0]
            purchase = float(line[11])
            redeem = float(line[12])
            direct_p = float(line[13])
            share_p = float(line[14])
            consume_r = float(line[15])
            transfer_r = float(line[16])
            yield1 = float(line[1])
            yield7 = float(line[2])
            interest_O_N = float(line[3])
            interest_1_W = float(line[4])
            interest_2_W = float(line[5])
            interest_1_M = float(line[6])
            interest_3_M = float(line[7])
            interest_6_M = float(line[8])
            interest_9_M = float(line[9])
            interest_1_Y = float(line[10])
            X.append([purchase, redeem, yield1, yield7, interest_O_N, interest_1_W, interest_2_W, interest_1_M,\
                    interest_3_M, interest_6_M, interest_9_M, interest_1_Y, \
                    direct_p, share_p, consume_r, transfer_r])
    if reverse:
        X.reverise()
    X = np.array(X)
    mins = []
    maxs = []
    if norm == 'minmax':
        mins = X.min(axis=0, keepdims=True) - 1e-4
        maxs = X.max(axis=0, keepdims=True) + 1e-4
        X = (X-mins) / (maxs - mins)
        X[X>1] = 0
        X = X*2 -1
    
    if maxlen is None:
        maxlen = (X.shape[0]-1)
    maxlen = min(maxlen+1, X.shape[0])/step
    assert(maxlen > 1)

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
    Y = data[:, 1:, :]
    if norm == 'minmax':
        return (X, Y, mins, maxs) # [timerange/maxlen, maxlen, 2+2+8+4]
    else:
        return (X,Y)

def load_data(csv_path, norm='minmax', sz=None, maxlen=None, step=1, reverse=False):
    data = generate_data(csv_path, norm, sz, maxlen, step, reverse)
    return data
