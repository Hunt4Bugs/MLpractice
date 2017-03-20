import pandas as pd
import numpy as np
from numpy.linalg import inv

def computeCost(X, y, theta):
    m = len(y)
    x = X.reshape(m, 2)
    h_vec = np.dot(x,theta).reshape(m, 1)
    return ((sum((h_vec-y)**2))/(2*m))[0]

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        temp0 = theta[0] - (alpha/m) * np.dot((np.dot(X, theta) - y).T, X[:,0])[0]
        temp1 = theta[1] - (alpha/m)*np.dot((np.dot(X,theta) - y).T,X[:,1])[0]
        theta[0] = temp0
        theta[1] = temp1

def gdMulti(x, y, theta):
    X = x.reshape(len(y), -1)
    a = inv(np.dot(X.T, X))
    b = np.dot(a, X.T)
    return np.dot(b,y)
