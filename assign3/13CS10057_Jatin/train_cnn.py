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
import os

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D 
from keras.models import load_model, Sequential
from keras.optimizers import SGD


def fetch_model(modelfile):
    if not os.path.exists(modelfile):
        print 'Downloading model from my github repository ...'
        os.system('wget -P weights https://raw.github.com/jatinarora2702/DeepLearning/master/assign3/13CS10057_Jatin/weights/modelparams_cnn.hdf5')
        print 'Download Complete.'


def train(trainX, trainY):
    model = Sequential()
    model.add(Conv2D(24, (3, 3), activation='relu', input_shape=trainX.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=30, batch_size=100)
    model.save('weights/modelparams_cnn.hdf5')    


def test(testX):
    fetch_model('weights/modelparams_cnn.hdf5')
    model = load_model('weights/modelparams_cnn.hdf5')
    bsize = testX.shape[0]
    out = model.predict(testX, batch_size=bsize, verbose=1)
    out = np.argmax(out, axis=1)
    return out
