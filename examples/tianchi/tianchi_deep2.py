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
from keras.datasets import tianchi
import numpy as np

batch_size = 128
nb_epoch = 1000
norm = 'minmax'
nblock = 4
hidden_units = 256
step = 1
nb_sample = 100
test_days = 31
train_days = 427 - test_days*2
tg=-1
train_split = 0.8
features= [0,1]

np.random.seed(1337) # for reproducibility

def load_data(sz, train_split, norm, step, features):
    # the data, shuffled and split between tran and test sets
    (X, Y, mins, maxs) = tianchi.load_data(csv_path='/home/zhaowuxia/dl_tools/datasets/tianchi/total_itp_pca2.csv', norm = norm, sz = sz, maxlen = None, step=step, reverse=False) 
    print(X.shape, Y.shape)

    sz = X.shape[0]

    train_sz = max(1, int(sz * train_split))
    X_train = X[:train_sz, :, features]
    y_train = Y[:train_sz, :, features]
    X_test = X[train_sz:, :, features]
    y_test = Y[train_sz:, :, features]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    mins = mins[:, features]
    maxs = maxs[:, features]
    print(np.fabs(y_train - X_train).mean(), np.fabs(y_test - X_test).mean())
    return (X_train, y_train, X_test, y_test, mins, maxs)

def build_model():
    model = Sequential()
    #model.add(Embedding(bins, 256))
    model.add(DEEPLSTM(input_dim=len(features), output_dim=hidden_units,init='glorot_normal',  num_blocks=nblock, return_seq_num=0, truncate_gradient=tg))
    #model.add(LSTM(input_dim=hidden_units, output_dim=hidden_units,init='glorot_normal',  return_sequences=True, truncate_gradient=tg))
    #model.add(Dropout(0.5))
    #model.add(LSTM(input_dim=hidden_units, output_dim=hidden_units,init='glorot_normal',  return_sequences=True, truncate_gradient=tg))
    #model.add(Dropout(0.5))
    model.add(TimeDistributedDense(hidden_units*nblock, len(features)))
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
        for j in range(len(features)):
            f.write('pred,gnd,')
        f.write('\n')
        for i in range(T):
            if i >= len(gnd):
                for j in range(len(features)):
                    f.write('%.4f,0,'%pred[i][j])
                f.write('\n')
            else:
                for j in range(len(features)):
                    f.write('%.4f,%.4f,'%(pred[i][j], gnd[i][j]))
                f.write('\n')

def write_ans(save_path, pred):
    print(pred.shape)
    T = pred.shape[0]
    with open(save_path, 'w') as f:
        for i in range(T):
            f.write('201409%02d,%d,%d\n'%(i+1, pred[i][0], pred[i][1]))

def recurrent_predict(model, x_history, pred_step, return_sequences=True):
    # x_history : [nb_sample, T, 1]
    # pred_step : int
    print('Predicting...')
    print(x_history.shape, pred_step)
    T = x_history.shape[1]
    nb_samples = x_history.shape[0]
    x = np.zeros([nb_samples, T+pred_step, len(features)])
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
              
def compute_loss(gnd, pred):
    # gnd: [T, k]
    # pred: [T, k]
    error = np.fabs(gnd-pred)/gnd
    mean_error = error.mean(0)
    for i in mean_error:
        print('%.4f'%i)
    return mean_error

if __name__=='__main__':
    (X_train, y_train, X_test, y_test, mins, maxs) = load_data(nb_sample, train_split, norm, step, features)
    X = X_test.copy().mean(0, keepdims=True)
    y = y_test.copy().mean(0, keepdims=True)
    X_train = X_train[:, :train_days]
    y_train = y_train[:, :train_days]
    X_test = X_test[:, :train_days+test_days]
    y_test = y_test[:, :train_days+test_days]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    #write_csv('csv2/train2_1deeplstm4__sz%d.csv'%(nb_sample), X[0], X[0])

    model = build_model()
    #model.load_weights('models2/2fea/train2_1ddeplstm4_%d_model_mae_sz%d_%d'%(hidden_units, nb_sample, train_days))
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test), save_path='models2/2fea/train2_1deeplstm4_%d_model_mae_sz%d_%d'%(hidden_units, nb_sample, train_days))
    model.save_weights('models2/2fea/train2_1deeplstm4_%d_model_mae_sz%d_%d_final'%(hidden_units, nb_sample, train_days), overwrite=True)
    model.load_weights('models2/2fea/train2_1deeplstm4_%d_model_mae_sz%d_%d'%(hidden_units, nb_sample, train_days))

    score = model.evaluate(X, y, batch_size=batch_size)
    print('Test score:', score)

    gnd = np.concatenate((X, y[:,-1:,:]), axis=1).mean(0, keepdims=True)
    pred1 = recurrent_predict(model, X[:, :train_days+test_days], 2*test_days, return_sequences=True)
    write_csv('csv2/2fea/train2_1deeplstm4_%d_mae_%d_%d_%d.csv'%(hidden_units, nb_sample, train_days+test_days, 2*test_days), gnd[0], pred1[0])
    pred2 = recurrent_predict(model, X[:, :train_days/2], train_days/2+2*test_days, return_sequences=True)
    write_csv('csv2/2fea/train2_1deeplstm4_%d_mae_%d_%d_%d.csv'%(hidden_units, nb_sample, train_days/2, train_days/2+2*test_days), gnd[0], pred2[0])

    gndo = (gnd[0]+1)/2*(maxs-mins)+mins
    pred = (pred1[0]+1)/2*(maxs-mins)+mins
    print('T-1 loss:')
    compute_loss(gndo[train_days+test_days:], gndo[train_days:train_days+test_days])
    print('lstm loss:')
    compute_loss(gndo[train_days+test_days:], pred[train_days+test_days:train_days+2*test_days])
    
    pred3 = recurrent_predict(model, gnd, 30, return_sequences=True)
    pred3 = (pred3[0]+1)/2*(maxs-mins)+mins
    pred3 = pred3.round().astype(int)
    write_ans('csv2/2fea/train2_1deeplstm4_%d_tc_comp_predict_table.csv'%(hidden_units), pred3[-30:])

