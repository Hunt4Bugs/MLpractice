from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, Dropout
from keras.layers import Activation, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import SGD
import pandas as pd
import numpy as np


#get data
print "Loading Data"
dat = pd.read_csv('train.csv', sep=',', header=0).values
"""finaldat = pd.read_csv('test.csv', sep=',', header=0).values
trainx = np.array([np.array(i[1::]) for i in dat])/255
trainx = trainx.reshape(trainx.shape[0], 1, 28, 28).astype('float32')
finalx = np.array([np.array(i[::]) for i in finaldat])/255
finalx = finalx.reshape(finalx.shape[0], 1, 28, 28).astype('float32')
trainy = np.array([i[0] for i in dat])"""
(trainx,trainy),(testx,testy) = mnist.load_data()
trainx = trainx.reshape(-1, 1, 28, 28)
testx = testx.reshape(-1, 1, 28, 28)
print trainx.shape

#cnn model
filters=32
pool = 2
conv = 3
model = Sequential([
	Convolution2D(filters,conv,conv, border_mode='same', input_shape=(1,28,28)),
	Activation('relu'),
	MaxPooling2D(pool_size=(pool,pool)),
	Convolution2D(filters,conv,conv),
	Activation('relu'),
	MaxPooling2D(pool_size=(pool,pool)),
	Flatten(),
	Dense(128),
	Activation('relu'),
	Dropout(0.5),
	Dense(10),
	Activation('softmax')
])

epochs = 64
lrate=0.01
sgd=SGD(lr=lrate,momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

#fit and evaluate model
"""testx = trainx[int(len(trainx)*0.7):]
trainx = trainx[:int(len(trainx) * 0.7)]
testy = trainy[int(len(trainy) * 0.7):]
trainy = trainy[:int(len(trainy) * 0.7)]"""
trainy = np_utils.to_categorical(trainy, 10)
testy = np_utils.to_categorical(testy,10)
model.fit(trainx,trainy, validation_data=(testx,testy), nb_epoch=epochs, batch_size=128,verbose=1)
scores=model.evaluate(testx,testy, verbose=0)
print ("\nAccuracy: " + str(scores[1]*100))
