from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

#get data
print "Loading Data"
dat = pd.read_csv('train.csv', sep=',', header=0).values
finaldat = pd.read_csv('test.csv', sep=',', header=0).values
finalx = np.array([i[::] for i in finaldat])
trainx = np.array([i[1::] for i in dat])
trainy = np.array([i[0] for i in dat])
ty = []
for i in range(len(trainy)):
	test = [0.0] * 10
	test[int(trainy[i])] = 1.0
	ty.append(test)
trainy = np.array(ty)

testx = trainx[int(len(trainx) * .7)::]
testy = trainy[int(len(trainy) * .7)::]

#declare and train model
print "Training Network"
model = Sequential([
	Dense(512, input_dim=784),
	Activation('sigmoid'),
	Dense(24),
	Activation('sigmoid'),
	Dense(10),
	Activation('sigmoid')
])

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(trainx, trainy, batch_size=100, nb_epoch=5, validation_data=(testx, testy))
evals = model.evaluate(testx, testy, verbose=0)
print "\nTest score:" + str(model.metrics_names)
print "predicting"
results = model.predict_classes(finalx, batch_size=32)
print results
