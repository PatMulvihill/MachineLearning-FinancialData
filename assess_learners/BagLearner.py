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

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

        SIZE = self.Xtrain.shape[0]
        index = [np.random.random_integers(0, SIZE - 1, size=SIZE) for i in range(self.bags)]
        self.bags_x = [self.Xtrain[index[i]] for i in range(self.bags)]
        self.bags_y = [self.Ytrain[index[j]] for j in range(self.bags)]

    def query(self,Xtest):

        # depend on the model we built, should returns a predicted value

        learners = []
        bag_res = np.zeros((len(self.bags_x),len(Xtest),))

        self.Xtest = Xtest

        for i in range(0,self.bags):
            learners.append(self.learner(**self.kwargs))

        for j in range(0,len(learners)):
            learners[j].addEvidence(self.bags_x[j],self.bags_y[j])
            bag_res[j] = learners[j].query(self.Xtest)

        return np.mean(bag_res, axis=0)


if __name__=="__main__":
    print "BagLearner"