from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import stock_one
from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, Merge
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import DEEPLSTM, GRU
from keras.utils import np_utils
from keras.objectives import to_categorical, categorical_crossentropy
from keras.datasets import stock_one
import numpy as np

batch_size = 128
nb_epoch = 1000
norm = 'minmax'
nblocks = 4
hidden_units = 256
#maxlen = 100
step = 10
train_days = 2500/step
test_days = 500/step
nb_sample = 100
tg=-1
#train_split = 0.8
stock_num = 'pkl_tables/600674'

#np.random.seed(1337) # for reproducibility

def load_data(sz, maxlen, stock_num, norm, step):
    # the data, shuffled and split between tran and test sets
    (X, Y, mins, maxs) = stock_one.load_data('/home/zhaowuxia/dl_tools/datasets/stock/%s.pkl'%(stock_num), norm = norm, sz = sz, maxlen = None, step = step, reverse=True ) 
    print(X.shape, Y.shape, mins.shape, maxs.shape)

    sz = X.shape[0]
    maxlen = min(maxlen, X.shape[1])

    X = X[:, -maxlen:, 3].reshape(sz, maxlen, 1)
    Y = np.concatenate((X[:, 1:, :], Y[:, 3].reshape(sz, 1, 1)), axis=1)
    mins = mins[:, 3]
    maxs = maxs[:, 3]
    return (X, Y, mins, maxs)

def build_model():
    model = Sequential()
    model.add(DEEPLSTM(input_dim=1, output_dim=hidden_units,init='glorot_normal', num_blocks=nblocks, return_seq_num=0, truncate_gradient=tg))
    #model.add(Dropout(0.5))
    model.add(TimeDistributedDense(hidden_units*nblocks, 1))
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
    (X, Y, mins, maxs) = load_data(nb_sample, train_days+test_days, stock_num, norm, step)
    X_train = X[:, :train_days]
    y_train = X[:, :train_days]
    X_test = X[:, :train_days+test_days/2]
    y_test = Y[:, :train_days+test_days/2]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, mins.shape, maxs.shape)

    model = build_model()
    #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test,  y_test), save_path='models/test/1deeplstm4_256_%s_%d_%d'%(stock_num.split('/')[-1], nb_sample, train_days))
    #model.save_weights('models/test/1deeplstm4_256_%s_%d_%d_final'%(stock_num.split('/')[-1], nb_sample, train_days), overwrite=True)
    model.load_weights('models/test/1deeplstm4_256_%s_%d_%d'%(stock_num.split('/')[-1], nb_sample, train_days))

    gnd = X.mean(0, keepdims=True)
    pred1 = recurrent_predict(model, gnd[:, :train_days], 2*test_days, return_sequences=True)
    write_csv('csv/test/1deeplstm4_256_%s_%d_%d_%d.csv'%(stock_num.split('/')[-1], nb_sample, train_days, 2*test_days), gnd[0], pred1[0])
    pred2 = recurrent_predict(model, gnd[:, :train_days/2], train_days/2 + test_days, return_sequences=True)
    write_csv('csv/test/1deeplstm4_256_%s_%d_%d_%d.csv'%(stock_num.split('/')[-1], nb_sample, train_days/2, train_days/2+2*test_days), gnd[0], pred2[0])

    pred = recurrent_predict(model, gnd[:, :train_days+test_days/2], test_days/2, return_sequences=True)
    pred = (pred[0]+1)/2*(maxs-mins)+mins
    gnd = (gnd[0]+1)/2*(maxs-mins)+mins
    for step in range(5, test_days/2+1, 5):
        error1 = compute_loss(gnd[train_days+test_days/2:train_days+test_days/2+step], pred[train_days+test_days/2:train_days+test_days/2+step])
    error2 = compute_loss(gnd[train_days+test_days/2:train_days+test_days/2+step], gnd[train_days+test_days/2-step:train_days+test_days/2])
    print('predict step = ', step, ': mean relative loss = ', error1, ', T-1 loss = ', error2)
