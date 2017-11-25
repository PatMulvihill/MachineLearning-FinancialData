import numpy as np
from random import randint


class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return 'lwang496'

    def get_indexes(self, Xtrain):
        # get the index for the left tree, right tree, get split index and split value
        all_index = []
        index = randint(0, Xtrain.shape[1] - 1)
        index1 = randint(0, Xtrain.shape[0] - 1)
        index2 = randint(0, Xtrain.shape[0] - 1)
        randomn1 = Xtrain[index1][index]
        randomn2 = Xtrain[index2][index]
        split_value = (randomn1 + randomn2) / 2
        left_index = []
        right_index = []
        for i in xrange(Xtrain.shape[0]):
            if Xtrain[i][index] <= split_value:
                left_index.append(i)
            else:
                right_index.append(i)
        all_index.append(left_index)
        all_index.append(right_index)
        all_index.append(index)
        all_index.append(split_value)
        return all_index

    def build_tree(self, Xtrain, Ytrain):
        # builder the tree based on the Xtrain and Ytrain
        if Xtrain.shape[0] < 1:
            return np.array([-1, -1, None, None])

        if len(np.unique(Ytrain)) == 1:
            return np.array([-1, Ytrain[0], None, None])

        if Xtrain.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(Ytrain), None, None])

        left_index, right_index, split_index, split_value = [],[],[],[]

        while len(left_index) < 1 or len(right_index) < 1:
            all_indexes = self.get_indexes(Xtrain)
            left_index, right_index, split_index, split_value = all_indexes[0],all_indexes[1],all_indexes[2],all_indexes[3]

        left_tree = self.build_tree(np.array([Xtrain[i] for i in left_index]), np.array([Ytrain[i] for i in left_index]))
        right_tree = self.build_tree(np.array([Xtrain[i] for i in right_index]), np.array([Ytrain[i] for i in right_index]))

        if len(left_tree.shape) == 1:
            root = [split_index, split_value, 1, 2]
        else:
            root = [split_index, split_value, 1, left_tree.shape[0] + 1]
        return np.vstack((root, left_tree, right_tree))

    def addEvidence(self, Xtrain, Ytrain):
        # return the tree I built
        self.tree = self.build_tree(Xtrain, Ytrain)

    def query(self, Xtest):
        # traverse the tree, find the leaf
        result = []
        for each_test in Xtest:
            row = 0
            i = int(self.tree[row][0])
            while i >= 0:
                val = self.tree[row][1]
                if each_test[i] <= val:
                    row = row + int(self.tree[row][2])
                else:
                    row = row + int(self.tree[row][3])
                i = int(self.tree[row][0])
            result.append(self.tree[row][1])
        return result




