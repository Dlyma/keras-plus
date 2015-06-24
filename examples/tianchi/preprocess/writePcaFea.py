import os
import cPickle
import numpy as np
from keras.datasets import tianchi

pcaVec = cPickle.load(open('res/pcaVecAll'))
pcaMean = cPickle.load(open('res/pcaMeanAll'))

def write_csv(save_path, data):
    # data: [T, 12]
    T = data.shape[0]
    with open(save_path, 'w') as f:
        for i in range(data.shape[1]):
            f.write('fea%d,'%i)
        f.write('\n')
        for i in range(T):
            for j in data[i]:
                f.write('%.4f,'%j)
            f.write('\n')
    
(X, Y) = tianchi.load_data(csv_path='/home/zhaowuxia/dl_tools/datasets/tianchi/total_itp.csv', norm='')
data = np.vstack((X[0,:,2:], Y[0, -1, 2:]))
nSmp, nFea = data.shape
data = data - np.tile(pcaMean, (nSmp, 1))
data = np.dot(data, pcaVec)
print(data.shape)

write_csv('res/pca_feature.csv', data)
