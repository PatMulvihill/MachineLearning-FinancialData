import numpy as np
import LinRegLearner as lrl
from random import randint

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        return 'lwang496'

    def addEvidence(self,Xtrain,Ytrain):
        # Randomly select the set of data
        index = [np.random.random_integers(0, Xtrain.shape[0] - 1, Xtrain.shape[0])]
        self.Xbags = [Xtrain[index[i]] for i in range(self.bags)]
        self.Ybags = [Ytrain[index[j]] for j in range(self.bags)]

    def query(self,Xtest):

        result = []
        learners = []

        for i in range(0,self.bags):
            learners.append(self.learner(**self.kwargs))

        for i in range(0,len(learners)):
            learners[i].addEvidence(self.Xbags[i],self.Ybags[i])
            result.append(learners[i].query(Xtest))

        if self.verbose:
            print np.mean(np.array(result), axis=0)

        return np.mean(np.array(result), axis=0)


