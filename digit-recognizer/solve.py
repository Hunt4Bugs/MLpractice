from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import SGD
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
	Dropout(0.2),
	Dense(10),
	Activation('sigmoid')
])

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.fit(trainx, trainy, batch_size=100, nb_epoch=15, validation_data=(testx, testy))
evals = model.evaluate(testx, testy, verbose=0)
print "\nTest accuracy:" + str(evals[1])
print "predicting"
results = model.predict_classes(finalx, batch_size=32)
results_file = pd.DataFrame({'ImageId':[i+1 for i in range(len(results))], 'Label':results})
results_file.to_csv('results.csv', index=False)
print results_file
