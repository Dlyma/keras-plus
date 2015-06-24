from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, DEEPLSTM
from keras.datasets import imdb

'''
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage 
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes: 

    - RNNs are tricky. Choice of batch size is important, 
    choice of loss and optimizer is critical, etc. 
    Most configurations won't converge.

    - LSTM loss decrease during training can be quite different 
    from what you see with CNNs/MLPs/etc. It's more or less a sigmoid
    instead of an inverse exponential.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

    250s/epoch on GPU (GT 650M), vs. 400s/epoch on CPU (2.4Ghz Core i7).
'''

max_features=20000
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 128

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(path='/home/zhaowuxia/dl_tools/datasets/imdb.pkl', nb_words=max_features, test_split=0.2)

X_train = X_train[:1024]
y_train = y_train[:1024]
X_test = X_test[:1024]
y_test = y_test[:1024]


print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

nb = 2
hidden_units = 128
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(DEEPLSTM(256, hidden_units, return_seq_num=1, num_blocks=nb))
model.add(Dropout(0.5))
model.add(Dense(hidden_units*nb, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")


print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5, validation_split=0.1, show_accuracy=True, save_path='test_model')
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, y_test)
print('Test accuracy:', acc)

