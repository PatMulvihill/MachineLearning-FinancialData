import numpy as np
from random import randint


class RTLearner(object):

    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return 'lwang496'

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

    def get_indexes(self, x_train, num_instances):
        split_index = randint(0, x_train.shape[1] - 1)
        split_index1 = randint(0, num_instances - 1)
        split_index2 = randint(0, num_instances - 1)
        split_value = (x_train[split_index1][split_index]
                     + x_train[split_index2][split_index])/2
        left_index = [i for i in xrange(x_train.shape[0])
                        if x_train[i][split_index] <= split_value]
        right_index = [i for i in xrange(x_train.shape[0])
                         if x_train[i][split_index] > split_value]
        return left_index, right_index, split_index, split_value

    def build_tree(self, Xtrain, Ytrain):

        if Xtrain.shape[0] == 0:
            return np.array([-1, -1, -1, -1])
        if Xtrain.shape[0] == 1:
            return np.array([-1, Ytrain[0], None, None])
        if len(np.unique(Ytrain)) == 1:
            return np.array([-1, Ytrain[0], None, None])
        if Xtrain.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(Ytrain), -1, -1])

        # get the left, right indexes, the max index, and the split value

        left_index, right_index, split_index, split_value = self.get_indexes(Xtrain, Ytrain)

        left_Xtrain = np.array([Xtrain[i] for i in left_index])
        left_Ytrain = np.array([Ytrain[i] for i in left_index])
        right_Xtrain = np.array([Xtrain[i] for i in right_index])
        right_Ytrain = np.array([Ytrain[i] for i in right_index])

        left_tree = self.build_tree(left_Xtrain, left_Ytrain)
        right_tree = self.build_tree(right_Xtrain, right_Ytrain)
        root = [split_index, split_value, 1, left_tree.shape[0] + 1]
        return np.append(root, left_tree, right_tree)



    def traverse_tree(self, each_test, row):

        if self.tree[row][0] == -1:
            return self.tree[row][1]
        if each_test[self.tree[row][0]] <= self.tree[row][1]:
            return self.traverse_tree(each_test, row + int(self.tree[row][2]))
        else:
            return self.traverse_tree(each_test, row + int(self.tree[row][3]))

    def query(self, Xtest):
        result = []
        for each_test in Xtest:
            result.append(self.traverse_tree(each_test,0))
        return np.array(result)