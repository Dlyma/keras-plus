from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, Merge
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.utils import np_utils
from keras.datasets import stock_cluster
import numpy as np

batch_size = 128
nb_epoch = 1000
norm = 'minmax'
hidden_units = 256
#maxlen = 100
step = 10
train_days = 2000/step
test_days = 500/step
nb_sample = 100
tg=-1
train_split = 0.8
cluster_num = 1

np.random.seed(1337) # for reproducibility

def load_data(sz, maxlen, train_split, cluster_num, norm, step):
    # the data, shuffled and split between tran and test sets
    (X, Y, mins, maxs) = stock_cluster.load_data(csv_path='/home/zhaowuxia/dl_tools/datasets/stock/clusters/cluster%d.csv'%(cluster_num), norm = norm, sz = sz, maxlen = None, step = step, reverse=False, mean=True) 
    print(X.shape, Y.shape)

    sz = X.shape[0]
    maxlen = min(maxlen, X.shape[1])

    X = X[:, -maxlen:, 0].reshape(sz, maxlen, 1)
    Y = Y[:, 0].reshape(sz, 1)
    print(X.shape, Y.shape)

    train_sz = max(1, int(sz * train_split))
    X_train = X[:train_sz]
    y_train = np.zeros(X_train.shape)
    y_train[:, :X_train.shape[1]-1, :] = X_train[:, 1:,:]
    y_train[:, -1, :]  = Y[:train_sz]
    
    X_test = X[train_sz:]
    y_test = np.zeros(X_test.shape)
    y_test[:, :X_test.shape[1]-1, :] = X_test[:, 1:,:]
    y_test[:, -1, :]  = Y[train_sz:]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    print(np.square(X_train - y_train).mean(), np.square(X_test-y_test).mean())
    return (X_train, y_train, X_test, y_test, mins, maxs)

def build_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=hidden_units,init='glorot_normal',  return_sequences=True,truncate_gradient=tg))
    model.add(TimeDistributedDense(hidden_units, 1))
    #model.add(Activation('relu'))

    #sgd=SGD(lr=1e-3, momentum=0.95, nesterov=True, clipnorm=5.0)
    #rms = RMSprop(clipnorm=5.0)
    model.compile(loss='mae', optimizer='adam')
    return model

def write_csv(save_path, gnd, pred):
    # gnd: [T, 1]
    # pred: [T, 1]
    T = pred.shape[0]
    with open(save_path, 'w') as f:
        f.write('pred,gnd\n')
        for i in range(T):
            if i >= len(gnd):
                f.write('%.4f,0,\n'%pred[i])
            else:
                f.write('%.4f,%.4f,\n'%(pred[i], gnd[i]))

def recurrent_predict(model, x_history, pred_step, return_sequences=True):
    # x_history : [nb_sample, T, 1]
    # pred_step : int
    print('Predicting...')
    print(x_history.shape, pred_step)
    T = x_history.shape[1]
    nb_samples = x_history.shape[0]
    x = np.zeros([nb_samples, T+pred_step, 1])
    x[:, :T] = x_history
    y = []
    for i in range(pred_step):
        if i > 0 and i % 100 == 0:
            print('%d steps finishes'%i)
        y=model.predict(x[:, :T+i, :], verbose=0)
        if return_sequences:
            x[:, T+i, :] = y[:, T+i-1, :]
        else:
            x[:, T+i, :] = y.reshape(x[:, T+i, :].shape)
    if return_sequences:
        x[:, 1:T, :] = y[:, :T-1, :]
    print('Finish predicting')
    return x
              
def compute_loss(gnd, pred, verbose=False):
    error = np.fabs(gnd-pred)/gnd
    mean_error = error.mean(error.ndim-2)
    if verbose:
        for i in mean_error:
            print('%.4f'%i)
    return mean_error

if __name__=='__main__':
    (X_train, y_train, X_test, y_test, mins, maxs) = load_data(nb_sample, train_days+test_days, train_split, cluster_num, norm, step)
    X_train = X_train[:, :train_days]
    y_train = y_train[:, :train_days]
    X_test = X_test[:, :train_days]
    y_test = y_test[:, :train_days]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    #write_csv('csv/cluster/%s_%d_%d.csv'%(cluster_num, nb_sample, train_days+test_days), X[0], X[0])

    model = build_model()
    #model.load_weights('models/cluster/model_%smean_%d_%d'%(cluster_num, nb_sample, train_days))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test,  y_test), save_path='models/cluster/model_%smean_%d_%d'%(cluster_num, nb_sample, train_days))
    model.save_weights('models/cluster/1lstm256/model_%smean_%d_%d_final'%(cluster_num, nb_sample, train_days), overwrite=True)
    model.load_weights('models/cluster/model_%smean_%d_%d'%(cluster_num, nb_sample, train_days))

    (X, Y, mins, maxs) = stock_cluster.load_data(csv_path='/home/zhaowuxia/dl_tools/datasets/stock/clusters/cluster%d.csv'%(cluster_num), norm = norm, sz = None, maxlen = None, step = step, reverse=False, mean=False) 
    print('X:', X.shape, 'Y:', Y.shape)
    print('mins.shape:', mins.shape, 'maxs.shape:', maxs.shape)

    gnd = np.concatenate((X, Y.reshape(Y.shape[0], 1, 1)), axis=1)
    pred1 = recurrent_predict(model, X[:, :train_days], 2*test_days, return_sequences=True)
    pred2 = recurrent_predict(model, X[:, :train_days/2], train_days/2+2*test_days, return_sequences=True)
    for i in range(10):
        write_csv('csv/cluster/1lstm256_output_%smean_%d/%d_%d_num%d.csv'%(cluster_num, nb_sample, train_days, 2*test_days, i), gnd[i], pred1[i])
        write_csv('csv/cluster/1lstm256_output_%smean_%d/%d_%d_num%d.csv'%(cluster_num, nb_sample, train_days/2, train_days/2+2*test_days, i), gnd[i], pred2[i])

    pred = pred1[:, :train_days+test_days]
    #pred = recurrent_predict(model, X[:, :train_days], test_days, return_sequences=True)
    gnd = (gnd+1)/2*np.repeat(maxs-mins, gnd.shape[1], axis=1)+mins.repeat(gnd.shape[1], axis=1)
    pred = (pred+1)/2*np.repeat(maxs-mins, pred.shape[1], axis=1)+mins.repeat(pred.shape[1], axis=1)
    for i in range(10, test_days+1, 10):
        error = compute_loss(gnd[:, train_days:train_days+i], pred[:, train_days:train_days+i])
        print('predict step = ', i, ': mean rel. loss = ', error.mean(), ', min rel. loss = ', error.min(), ', max rel. loss = ', error.max())
        error = compute_loss(gnd[:, train_days:train_days+i], gnd[:, train_days-i:train_days])
        print('T-1 loss: mean rel. loss = ', error.mean(), ', min rel. loss = ', error.min(), ', max rel. loss = ', error.max())
