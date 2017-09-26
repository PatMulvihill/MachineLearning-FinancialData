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
    y4 =[]
    y5 = []

    for i in range(1,101):
        x1.append(i*2)
    for j in range (100):
        x2.append(randint(1, 19))
    for i in range(50):
        y1.append(i)
    for i in range(50):
        y2.append(1-i*2)


    matrix = [[x1[i], x2[i]] for i in range(100)]
    X= np.array(matrix)
    Y = np.hstack((np.array(y1),np.array(y2)))



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
