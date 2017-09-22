import numpy as np
import BagLearner as bag
import LinRegLearner as lrl
from random import randint

class InsaneLearner(object):
    def __init__(self, verbose):
        self.verbose = verbose
        self.learner = bag.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
    def author(self):
        return 'lwang496'
    def addEvidence(self, Xtrain, Ytrain):
        self.learner.addEvidence(Xtrain, Ytrain)
    def query(self,Xtest):
        result = self.learner.query(Xtest)
        if self.verbose:
            print np.mean(np.array(result), axis=0)
        return np.mean(np.array(result), axis=0)


