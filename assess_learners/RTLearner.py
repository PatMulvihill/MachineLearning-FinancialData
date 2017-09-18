import numpy as np
from random import randint


class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return 'lwang496'

    def get_indexes(self, x_train, num_instances):
        index = randint(0, x_train.shape[1] - 1)
        index1 = randint(0, num_instances - 1)
        index2 = randint(0, num_instances - 1)
        split_value = (x_train[index1][index] + x_train[index2][index]) / 2
        left_index = []
        right_index = []
        for i in xrange(x_train.shape[0]):
            if x_train[i][index] <= split_value:
                left_index.append(i)
            else:
                right_index.append(i)

        return left_index, right_index, index, split_value

    def build_tree(self, Xtrain, Ytrain):

        if Xtrain.shape[0] == 0:

            return np.array([-1, -1, -1, -1])
        if Xtrain.shape[0] <= self.leaf_size:

            return np.array([-1, np.mean(Ytrain), -1, -1])

        Y_values = np.unique(Ytrain)
        if len(Y_values) == 1:

            return np.array([-1, Ytrain[0], -1, -1])

        left_index, right_index, index, split_value =  self.get_indexes(Xtrain, Xtrain.shape[0])

        while len(left_index) < 1 or len(right_index) < 1:
            left_index, right_index, index, split_value = \
                self.get_indexes(Xtrain, Xtrain.shape[0])

        left_x_train = np.array([Xtrain[i] for i in left_index])
        left_y_train = np.array([Ytrain[i] for i in right_index])
        right_x_train = np.array([Xtrain[i] for i in right_index])
        right_y_train = np.array([Ytrain[i] for i in right_index])

        left_tree = self.build_tree(left_x_train, left_y_train)
        right_tree = self.build_tree(right_x_train, right_y_train)
        if len(left_tree.shape) == 1:
            num_left = 2
        else:
            num_left = left_tree.shape[0] + 1
        root = [index, split_value, 1, num_left]
        return np.vstack((root, np.vstack((left_tree, right_tree))))

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

    def traverse(self, instance, row=0):
        index = int(self.tree[row][0])
        if index == -1:
            return self.tree[row][1]
        if instance[index] <= self.tree[row][1]:
            return self.traverse(instance, row + int(self.tree[row][2]))
        else:
            return self.traverse(instance, row + int(self.tree[row][3]))

    def query(self, Xtest):
        result = []
        for instance in Xtest:
            result.append(self.traverse(instance))
        return np.array(result)