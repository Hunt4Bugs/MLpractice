import pandas as pd
import numpy as np
from scipy.optimize import minimize

g = lambda z : 1/(1 + np.exp(-z))
h = lambda x, theta : g(x.dot(theta))

def mapFeatures(x, y): #x1,x2):
    ini_theta = np.array([[15],[15],[15]])
    reg = 0
    return minimize(dJ, ini_theta, args=(x,y,reg), method=None, jac=J, options={'maxiter':5000})
    """degree = 6
    out = [1]
    for i in range(degree+1):
        for j in range(i+1):
            out.append((x1 ** (i - j)).T.dot(x2 ** j))

    return np.array(out).T.flatten()"""

def dJ(theta, x, y, l):
    m = len(y)
    n = theta.shape[0]
    sums = np.zeros(n)
    delta = h(x, theta) - y
    return (1/(2. * m)) * np.sum(np.square(delta)) + (l/(2. * m)) * np.sum(np.square(theta))

def J(theta, x, y, l):
    m = len(y)
    print x.shape
    return (1/m) * x.dot(h(x, theta) - y) + (l/m) * theta

def gd(x, y):
    theta = np.zeros((3,1))
    X = np.ones((len(y), 3))

    for i in range(len(x)):
        temp = [1.0]
        for j in range(len(x[i])):
            temp.append(x[i][j])
        X[i] = temp

    a = np.array([1.0, 45.0, 85.])
    reg = np.ones((3,1))
    mapped = mapFeatures(X,y) #X[:, 0], X[:, 1])
    print mapped.X.shape
    #mapped = mapped.reshape((len(mapped), 1))
    mapped = h(mapped, theta)
    #print mapped.shape
    #print J(theta, mapped, y, 1)

    #theta = minimize(J, theta, args=(X, y), method=None, jac=dJ, options={'maxiter':1500}).x.T
    #print 'Prediction AFTER gradient descent : ' + str(h(a, theta))

def main():
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    x,y = data[:,:2], data[:,2]
    x,y = x.reshape(len(x), -1), y.reshape(len(y), 1)
    gd(x,y)

main()
