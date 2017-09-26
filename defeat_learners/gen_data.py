"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math
from random import randint

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4DT(seed=189):
    np.random.seed(seed)


    y1=[]
    y2=[]
    y3=[]
    y4 =[]
    y5 = []

    np.random.seed(seed)
    x1 = abs(np.random.random(size=(20, 3)))
    x2 = x1*10
    x3 = x2 *5
    x4 = 0.1 - x2
    x5 = 0.1 - x3
    X = np.vstack((x1,x2,x3,x4,x5))

    for i in range(50):
        y1.append(i)
    for i in range(50):
        y2.append(1 - i * 2)

    Y = np.hstack((np.array(y1), np.array(y2)))


    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    return X, Y

def best4LinReg(seed=14896833):
    np.random.seed(seed)
    X = np.random.random(size=(100,3)) * 200 - 50
    l = []
    for i in range(np.shape(X)[0]):
        l.append(X[i][0]*2 + X[i][1]*3 + X[i][2])

    Y = np.array(l)
    return X, Y

def author():
    return 'lwang496' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
