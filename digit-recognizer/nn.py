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

        self.ai = np.zeros((ni,1))
        self.ah = np.zeros((nh,1))
        self.ao = np.zeros((no,1))

        self.wh = np.random.rand(self.nh, self.ni+1)
        self.wo = np.random.rand(self.no, self.nh+1)

        self.ch = np.zeros((self.nh, self.ni))
        self.co = np.zeros((self.no, self.nh))

    def ff(self, x, b=1.0):
        self.ai = np.concatenate((x, np.array([b])), axis=0)
        #hidden activations
        self.wh[:,len(self.wh)-1] = np.ones(len(self.wh))
        t = self.wh.dot(self.ai)
        self.ah = np.concatenate((sigmoid(t), np.array([b])))

        #output activations
        self.wo[:,len(self.wo)-1] = np.ones(len(self.wo))
        a = self.wo.dot(self.ah)
        self.ao = np.sigmoid(a)
        return self.ao

    def bp(self, x, y, wh_grad, wo_grad):
        #N = learning rate
		#M = Momentum
        nn_out = self.ff(x)

        #output layer errors
        dzj = np.zeros((self.no,1))#[0.0] * self.no
        outerror = y.T - self.ao
        dzj = dsigmoid(self.ao) * outerror
        #for i in range(self.no):
       # 	error = y[i]-self.ao[i]
       # 	dzj[i] = dsigmoid(self.ao[i]) * error

        #hidden layer errors
        dyi = np.zeros((self.nh,1))#[0.0] * self.nh
        herror = dzj.dot(self.wo)
        dyi = dsigmoid(self.ah[0:self.nh:]) * herror[0:self.nh]
        #for i in range(self.nh):
        #    error = 0.0
        #    for j in range(self.no):
        #        error += dzj[j] * self.wh[i][j]
        #    dyi[i] = dsigmoid(self.ah[i]) * error

        #update weights output first, then hidden
        ah = self.ah.reshape(len(self.ah),1).T
        dzj = dzj.reshape(len(dzj), 1)
        wo_grad += dzj.dot(ah)
        """for i in range(self.no):
            for j in range(self.nh):
				change = dzj[i] * self.ah[j]
				self.wo[i][j] -= 0.01 * change #+ M * self.co[i][j]
				self.co[i][j] = change"""

        ai = self.ai.reshape(len(self.ai),1).T
        dyi = dyi.reshape(dyi.shape[0], 1)
        wh_grad += dyi.dot(ai)
        """for i in range(self.nh):
            for j in range(self.ni):
				change = dyi[i] * x[j]
				self.wh[i][j] = 0.01 * change #+ M * self.ch[i][j]
				self.ch[i][j] = change"""

    def train(self, x, y, alpha=0.0005, epochs=24, batch=100):
        print("Training Network")
        xtrain = x[:int(len(x)*.7)]
        ytrain = y[:int(len(y)*.7)]
        xtest = x[int(len(x)*.7)::]
        ytest = y[int(len(x)*.7)::]
        for i in range(epochs):
			print (str(i+1) + "/" + str(epochs) + " Epochs")
			wh_grad = np.zeros((self.nh,self.ni+1))
			wo_grad = np.zeros((self.no,self.nh+1))
			for i in range(len(xtrain)):
				print (str(i) + "/" + str(len(xtrain)), end="\r")
				self.bp(xtrain[i], ytrain[i], wh_grad, wo_grad)
			self.wh += -1 * (1/len(xtrain)) * np.multiply(alpha, wh_grad)
			self.wo += -1 * (1/len(xtrain)) * np.multiply(alpha, wo_grad)
			#get batch size and bp
			"""mini_batches = [xtrain[i:i+batch] for k in range(0, len(xtrain), batch)]
			mini_y = [ytrain[i] for i in range(0, len(ytrain), batch)]
			for mini_batch in mini_batches:
				self.batch_calculate(mini_batch, mini_y)"""
        print (self.wh)
        print (self.ao)
        np.savetxt('wh.o', self.wh)
        np.savetxt('wo.o', self.wo)
        right = wrong = 0
        for i in range(len(xtest)):
			out = self.ff(xtest[1])
			print (np.argmax(out))
			print (np.argmax(ytest[1])) 

nn = NN(784, 24, 10)
print ("Loading Data")
dat = pd.read_csv('train.csv', sep=',', header=0).values
trainx = np.array([np.array(i[1::]).astype(float) for i in dat])
trainy = np.array([float(i[0]) for i in dat])

ty = []
for i in range(len(trainy)):
	test = [0.0] * 10
	test[int(trainy[i])] = 1.0
	ty.append(np.array(test))
trainy = np.array(ty)

nn.train(trainx, trainy, epochs=5)


