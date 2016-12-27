import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funcs import *

def plotData(x,y):
    plt.plot(x, y, 'rx')
    plt.show()

def gd(x,y):
    theta = np.zeros((3,1))
    X = np.ones((len(y), 3))

    for i in range(len(x)):
        temp = [1]
        for j in range(len(x[i])):
            temp.append(x[i][j])
        X[i] = temp

    theta = gdMulti(X, y, theta)#, alpha, iterations)

    a = np.array([1.0, 2200.0, 4.0])
    b = np.array([1.0, 3000.0, 3.0])

    predict1 = np.dot(a, theta)[0]
    predict2 = np.dot(b, theta)[0]

    print "Theta's found by gradient descent: " + str(theta[0]) + " " + str(theta[1])

    print "For population = 35,000, we predict a profit of " + str(predict1)
    print "For population = 70,000, we predict a profit of " + str(predict2)

def main():
    data = np.genfromtxt('ex1data2.txt', delimiter=',')
    x,y = data[:,:2], data[:,2]
    x,y = x.reshape(len(x),-1), y.reshape(len(y),1)
    #plotData(x,y)
    gd(x,y)

main()
