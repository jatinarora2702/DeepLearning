'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Jatin Arora
Roll No.: 13CS10057

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np

from keras.layers import Activation, Dense 
from keras.models import load_model, Sequential
from keras.optimizers import SGD


def train(trainX, trainY):
    inpdim = trainX.shape[1] * trainX.shape[2]
    trainX = trainX.reshape((-1, inpdim))
    
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=inpdim))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=60, batch_size=100)
    model.save('weights/modelparams.hdf5')


def test(testX):
    bsize = testX.shape[0]
    inpdim = testX.shape[1] * testX.shape[2]
    testX = testX.reshape((-1, inpdim))
    
    model = load_model('weights/modelparams.hdf5')
    out = model.predict(testX, batch_size=bsize, verbose=1)
    out = np.argmax(out, axis=1)
    return out
