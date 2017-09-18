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

        if Xtrain.shape[0] < 1:
            return np.array([-1, -1, None, None])

        if len(np.unique(Ytrain)) == 1:
            return np.array([-1, Ytrain[0], None, None])

        if Xtrain.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(Ytrain), None, None])

        left_index, right_index, split_index, split_value = [],[],[],[]

        while len(left_index) < 1 or len(right_index) < 1:
            left_index, right_index, split_index, split_value = self.get_indexes(Xtrain, Xtrain.shape[0])

        left_Xtrain = np.array([Xtrain[i] for i in left_index])
        right_Xtrain = np.array([Xtrain[i] for i in right_index])
        left_Ytrain = np.array([Ytrain[i] for i in left_index])
        right_Ytrain = np.array([Ytrain[i] for i in right_index])

        left_tree = self.build_tree(left_Xtrain, left_Ytrain)
        right_tree = self.build_tree(right_Xtrain, right_Ytrain)
        if len(left_tree.shape) == 1:
            num_left = 2
        else:
            num_left= left_tree.shape[0] + 1
        root = [split_index, split_value, 1, num_left]
        return np.vstack((root, np.vstack((left_tree, right_tree))))

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

    def traverse(self, each_test, row=0):
        index = int(self.tree[row][0])
        if index < 0:
            return self.tree[row][1]
        if each_test[index] <= self.tree[row][1]:
            return self.traverse(each_test, row + int(self.tree[row][2]))
        else:
            return self.traverse(each_test, row + int(self.tree[row][3]))

    def query(self, Xtest):
        i = 0
        sh = Xtest.shape[0]
        result = np.empty([sh])  # create an empty array
        while i < sh:
            arrayIndex = 0
            
            while ~np.isnan(self.tree[arrayIndex, 3]):  # checks if the row in "decision tree" array is a leaf
                val = self.tree[arrayIndex, 1]  # finds the split value
                if Xtest[i, int(self.tree[
                                     arrayIndex, 0])] <= val:  # compares the split value of the feature with the value of the same feature in test set
                    arrayIndex = arrayIndex + 1  # goes to the right ree
                else:
                    arrayIndex = arrayIndex + int(self.tree[arrayIndex, 3])  # goes to the left tree
            value = self.tree[arrayIndex, 1]
            result[i] = value
            i = i + 1
        return result