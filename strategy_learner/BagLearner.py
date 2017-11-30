import numpy as np
from collections import Counter

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.Xbags = []
        self.Ybags = []

    def author(self):
        return 'lwang496'

    def addEvidence(self,Xtrain,Ytrain):
        # Randomly select the set of data
        index = []
        for i in range(self.bags):
            index.append(np.random.random_integers(0, Xtrain.shape[0] - 1, Xtrain.shape[0]))

        for i in range(self.bags):
            self.Xbags.append(Xtrain[index[i]])
            self.Ybags.append(Ytrain[index[i]])

    def query(self,Xtest):

        result = []
        learners = []

        for i in range(0, self.bags):
            learners.append(self.learner(**self.kwargs))

        for i in range(0, len(learners)):
            # randomn here
            learners[i].addEvidence(self.Xbags[i], self.Ybags[i])
            result.append(learners[i].query(Xtest))

        if self.verbose:

            #need vote
            print np.min(np.array(result), axis=0)

        return np.min(np.array(result), axis=0)


