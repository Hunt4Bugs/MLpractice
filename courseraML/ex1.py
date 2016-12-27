import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funcs import *

def plotData(x,y):
    plt.plot(x, y, 'rx')
    plt.show()

def gd(x,y):
    X = np.ones((len(x), 2))
    theta = np.zeros((2,1))

    for i in range(len(x)):
        X[i][1] = x[i]

    iterations = 1500
    alpha = 0.01

    print 'Test Cost: ' + str(computeCost(X, y, theta))
    gradientDescent(X, y, theta, alpha, iterations)

    a = np.array([1.0, 3.5])
    b = np.array([1.0, 7.0])

    predict1 = np.dot(a, theta)[0]
    predict2 = np.dot(b, theta)[0]

    print "Theta's found by gradient descent: " + str(theta[0]) + " " + str(theta[1])

    print "For population = 35,000, we predict a profit of " + str(predict1*10000)
    print "For population = 70,000, we predict a profit of " + str(predict2*10000)

def main():
    data = np.genfromtxt('ex1data1.txt', delimiter=',')
    x,y = data[:,0], data[:,1]
    x,y = x.reshape(len(x),1), y.reshape(len(y),1)
    #plotData(x,y)
    gd(x,y)

main()
