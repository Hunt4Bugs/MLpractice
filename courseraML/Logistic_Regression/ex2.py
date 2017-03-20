import pandas as pd
import numpy as np
from scipy.optimize import minimize

g = lambda z : 1/(1 + np.exp(-z))
h = lambda x, theta : g(x.dot(theta))

def dJ(theta, x, y):
    m = len(y)
    n = theta.shape[0]
    sums = np.zeros(n)
    for i in range(n):
        sums[i] += np.dot((h(x, theta) - y).T, x[:,i])[0]/m
    return sums

def J(theta, x, y):
    m = len(y)
    J = -1. * (1./m) * (np.log(h(x, theta)).T.dot(y)[0] + np.log(1 - h(x,theta)).T.dot(1-y)[0])
    return J

def gd(x, y):
    theta = np.zeros((3,1))
    X = np.ones((len(y), 3))

    for i in range(len(x)):
        temp = [1.0]
        for j in range(len(x[i])):
            temp.append(x[i][j])
        X[i] = temp

    a = np.array([1.0, 45.0, 85.])

    theta = minimize(J, theta, args=(X, y), method=None, jac=dJ, options={'maxiter':1500}).x.T
    print 'Prediction AFTER gradient descent : ' + str(h(a, theta))

def main():
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    x,y = data[:,:2], data[:,2]
    x,y = x.reshape(len(x), -1), y.reshape(len(y), 1)
    gd(x,y)

main()
