from __future__ import print_function
import numpy as np
import pandas as pd

sigmoid = lambda z : 1 /(1+np.exp(-z))
dsigmoid = lambda x : x * (1-x)

#@param k: number of units in layer
class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni
        self.nh = nh
        self.no = no
        self.biases=np.ones((2,1))

        self.ai = np.zeros((ni,1))
        self.ah = np.zeros((nh,1))
        self.ao = np.zeros((no,1))

        self.wh = np.random.uniform(low=-1.0, high=1.0, size=(self.nh, self.ni))
        self.wo = np.random.uniform(low=-1.0, high=1.0, size=(self.no, self.nh))

        self.ch = np.zeros((self.nh, self.ni))
        self.co = np.zeros((self.no, self.nh))

    def ff(self, x, b=1.0):
        self.ai = x #np.concatenate((x, np.array([b])), axis=0)
        #hidden activations
        t = 1.0 + self.wh.dot(self.ai)
        self.ah = sigmoid(t)#sigmoid(np.concatenate((t, np.array([b]))))

        #output activations
        a = 1.0 + self.wo.dot(self.ah)
        self.ao = sigmoid( a )
        return self.ao

    def predict(self, x):
		results = []
		for i in x:
			results.append(np.argmax(self.ff(i)))
		return np.array(results)

    def evaluate(self, x, y):
		results = self.predict(x)
		print (results)
		loss = 0.0
		right = 0.0
		for i in range(len(results)):
			if results[i] == np.argmax(y[i]):
				right += 1.0
		accuracy = right / len(results)
		print ("accuracy : " + str(accuracy))

    def bp(self, x, y, wh_grad, wo_grad):
        nn_out = self.ff(x)

        #output layer errors
        outerror = y - self.ao
        dzj = dsigmoid(self.ao) * outerror
        dzj = dzj.reshape(len(dzj),1)

        #hidden layer errors
        herror = dzj.T.dot(self.wo)
        self.ah = ah = self.ah.reshape(len(self.ah),1)
        #dyi = dsigmoid(self.ah) * herror.T
        #dyi = dyi.reshape(len(dyi),1)
        dyi = np.zeros(self.nh)
        for i in range(self.nh):
			error = 0.0
			for  k in range(self.no):
				error += dzj[k] * self.wo[k][i]
			dyi[i] = dsigmoid(self.ah[i]) * error
        dyi = dyi.reshape(len(dyi),1)

        #weight gradients: output first, then hidden
        """print ('dzj : ',end='')
        print (dzj.shape)
        print ('dyi : ',end='')
        print (dyi.shape)
        print ('wh : ',end='')
        print (self.wh.shape)
        print ('wo : ',end='')
        print (self.wo.shape)"""
        wo_grad += dzj.dot(ah.T)
        self.wo += wo_grad

        ai = self.ai.reshape(len(self.ai),1).T
        wh_grad += dyi.dot(ai)
        self.wh += wh_grad

    def train(self, x, y, lr=0.01, epochs=24, batch=100):
        print("Training Network")
        xtrain = x[:int(len(x)*.7)]
        ytrain = y[:int(len(y)*.7)]
        xval = x[int(len(x)*.7):]#int(len(x) * .9):]
        yval = y[int(len(x)*.7):]#int(len(x) * .9):]
        #xtest = x[int(len(x) * .9):]
        #ytest = x[int(len(x) * .9):]

        for i in range(epochs):
			print (str(i+1) + "/" + str(epochs) + " Epochs ")
			for i in range(len(xtrain)):
				print (str(i) + "/" + str(len(xtrain)), end="\r")
				wh_grad = np.zeros((self.nh,self.ni))
				wo_grad = np.zeros((self.no,self.nh))
				self.bp(xtrain[i], ytrain[i], wh_grad, wo_grad)
				self.wh += -1 * (1/len(xtrain)) * np.multiply(lr, wh_grad)
				self.wo += -1 * (1/len(xtrain)) * np.multiply(lr, wo_grad)
			shuffle = np.random.permutation(xtrain.shape[0])
			xtrain,ytrain = xtrain[shuffle], ytrain[shuffle]

        np.savetxt('wh.o', self.wh)
        np.savetxt('wo.o', self.wo)
        self.evaluate(xval, yval)

nn = NN(784, 24, 10)
print ("Loading Data")
dat = pd.read_csv('train.csv', sep=',', header=0).values
testdat = pd.read_csv('test.csv', sep=',', header=0).values
trainx = np.array([np.array(i[1::]).astype(float) for i in dat])
trainy = np.array([float(i[0]) for i in dat])
testx = np.array([np.array(i).astype(float) for i in testdat])

ty = []
for i in range(len(trainy)):
	test = [0.0] * 10
	test[int(trainy[i])] = 1.0
	ty.append(np.array(test))
trainy = np.array(ty)
trainx /= 255.0
testx /= 255.0

nn.train(trainx, trainy, epochs=10)

results = nn.predict(testx)
results = pd.DataFrame({'ImageId':[i+1 for i in range(len(testx))], 'Label':results})
results.to_csv('results.csv', index=False)
