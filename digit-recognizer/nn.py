from __future__ import print_function
import numpy as np
import pandas as pd

sigmoid = lambda z : 1 /(1+np.exp(-z))
dsigmoid = lambda x : sigmoid(x) * (1-sigmoid(x))

#@param k: number of units in layer
class NN:
    def __init__(self, ni, nh, no, bias=1.0):
        self.ni = ni
        self.nh = nh
        self.no = no
        self.b = bias

        self.ah = [1.0] * nh
        self.ao = [1.0] * no

        self.wh = np.ones((self.nh, self.ni), dtype=np.float32)
        self.wo = np.ones((self.no, self.nh), dtype=np.float32)

        self.ch = np.zeros((self.nh, self.ni), dtype=np.float32)
        self.co = np.zeros((self.no, self.nh), dtype=np.float32)

    def ff(self, inp, b=1.0):
        #hidden activations
        t =  1.0 + self.wh.dot(inp)
        g = sigmoid(t)
        self.ah = g

        #output activations
        t = 1.0 + self.wo.dot(g)
        out = sigmoid(t)
        self.ao = out
        return out

    def bp(self, x, y, N=0.5, M=0.1):
        #N = learning rate
		#M = Momentum
        nn_out = self.ff(x)

        #output layer errors
        dzj = [0.0] * self.no
		#outerror = y - self.ao
		#dzj = dsigmoid(self.ao) * outerror
        for i in range(self.no):
            error = y[i]-self.ao[i]
            dzj[i] = dsigmoid(self.ao[i]) * error

        #hidden layer errors
        dyi = [0.0] * self.nh
		#herror = self.wh.dot(dzj)
		#dyi = digmoid(self.ah) * error
        for i in range(self.nh):
            error = 0.0
            for j in range(self.no):
                error += dzj[j] * self.wh[i][j]
            dyi[i] = dsigmoid(self.ah[i]) * error

        #update weights output first, then hidden
        for i in range(self.no):
            for j in range(self.nh):
				change = dzj[i] * self.ah[j]
				self.wo[i][j] += N * change + M * self.co[i][j]
				self.co[i][j] = change

        for i in range(self.nh):
            for j in range(self.ni):
				change = dyi[i] * x[j]
				self.wh[i][j] += N * change + M * self.ch[i][j]
				self.ch[i][j] = change

    def batch_calculate(self, mini_batch, y):
        for i in range(len(mini_batch)):
            self.bp(mini_batch[i],y[i])

    def train(self, x, y, epochs=24, batch=100):
        xtrain = x[:int(len(x)*.7)]
        ytrain = y[:int(len(y)*.7)]
        xtest = x[int(len(x)*.7)::]
        ytest = y[int(len(x)*.7)::]
        for i in range(epochs):
			print ("\r" + str(i+1) + "/" + str(epochs) + " Epochs")#, end="")
			for i in range(len(xtrain)):
				self.bp(xtrain[i], ytrain[i])
			#get batch size and bp
			"""mini_batches = [xtrain[i:i+batch] for k in range(0, len(xtrain), batch)]
			mini_y = [ytrain[i] for i in range(0, len(ytrain), batch)]
			for mini_batch in mini_batches:
				self.batch_calculate(mini_batch, mini_y)"""

        print (self.wh)
        #np.savetxt('wh.out', self.wh, fmt='%-10.6')
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        with open('wh.out', 'w') as f:
			f.write(np.array2string(self.wh, separator=','))
        print ("output weights")
        print (self.wo)
        with open('wo.out', 'w') as fi:
        	fi.write(np.array2string(self.wo, separator=','))
        #np.savetxt('wo.out', self.wo, fmt='%-10.6')

        for i in range(len(xtest)):
			out = self.ff(xtest[i])
			print (out) 
			print (str(ytest[i]))
			print (" ")
            
nn = NN(784, 24, 10)
print ("Loading Data")
dat = pd.read_csv('train.csv', sep=',', header=0).values
trainx = np.array([np.array(i[1::]) for i in dat])
trainy = np.array([float(i[0]) for i in dat])

ty = []
for i in range(len(trainy)):
	test = [0.0] * 10
	test[int(trainy[i])] = 1.0
	ty.append(np.array(test))
trainy = np.array(ty)

nn.train(trainx, trainy, epochs=1)


