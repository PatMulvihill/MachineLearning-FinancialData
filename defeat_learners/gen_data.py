"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math
from random import randint

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4DT(seed=1489):
    np.random.seed(seed)
    x1=[]
    x2=[]

    y1=[]
    y2=[]
    y3=[]
    for i in range(100):
        x1.append(i)
    for j in range (100):
        x2.append(j*5)
    for i in range(30):
        y1.append(randint(0, 9))
    for i in range(30):
        y2.append(randint(0, 9)*5)
    for i in range(40):
        y3.append(1-randint(0, 9)*10)
    matrix = [[x1[i], x2[i]] for i in range(100)]
    X= np.array(matrix)
    Y = np.hstack((np.array(y1),np.array(y2),np.array(y3)))



    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    return X, Y

def best4LinReg(seed=14896833):
    np.random.seed(seed)
    X = np.random.random(size=(100,3)) * 200 - 100
    l = []
    for i in range(np.shape(X)[0]):
        l.append(X[i][0]*2 + X[i][1]*3 + X[i][2])

    Y = np.array(l)
    return X, Y

def author():
    return 'lwang496' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
