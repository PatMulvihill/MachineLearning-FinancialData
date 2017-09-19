import numpy as np
import LinRegLearner as lrl

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
        index = []
        for i in range(self.bags):
            index.append[np.random.random_integers(0, Xtrain.shape[0] - 1, Xtrain.shape[0])]

        self.bags_x = [Xtrain[index[i]] for i in range(self.bags)]
        self.bags_y = [Ytrain[index[j]] for j in range(self.bags)]

    def query(self,Xtest):

        result = np.zeros((len(self.bags_x),len(Xtest),))
        learners = []

        for i in range(self.bags):
            learners.append(self.learner(**self.kwargs))

        for j in range(len(learners)):
            learners[j].addEvidence(self.bags_x[j],self.bags_y[j])
            result[j] = learners[j].query(Xtest)

        if self.verbose:
            print np.mean(result, axis=0)

        return np.mean(result, axis=0)


