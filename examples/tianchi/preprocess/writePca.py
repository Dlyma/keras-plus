import os
import cPickle
import numpy as np
from PCA import *
from keras.datasets import tianchi

(X, Y) = tianchi.load_data(csv_path='/home/zhaowuxia/dl_tools/datasets/tianchi/total_itp.csv', norm='')
print X.shape, Y.shape
data = np.vstack((X[0,:,2:], Y[0, -1, 2:]))

print "Finish loading data!"
print data.shape

vec, val, mea = PCA(data)
pcaVec = open('res/pcaVecAll','w')
pcaVal = open('res/pcaValAll','w')
pcaMean = open('res/pcaMeanAll','w')

cPickle.dump(vec, pcaVec, protocol = cPickle.HIGHEST_PROTOCOL)
cPickle.dump(val, pcaVal, protocol = cPickle.HIGHEST_PROTOCOL)
cPickle.dump(mea, pcaMean, protocol = cPickle.HIGHEST_PROTOCOL)

valsum = sum(val)
s = 0
f = open('res/valPercentage','w')
for i in range(len(val)):
    s += val[i]
    f.write(str(i) + ' ' + str(s/valsum) + '\n')

