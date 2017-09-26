import numpy as np

class DTLearner(object):

    def __init__(self, verbose = False):

        
        self.tree = np.array([])

    def author(self):
        return 'lwang496'

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

    def get_indexes(self, Xtrain, Ytrain):
        all_index = []
        index = -1;
        max_correlation = 0
        for i in range(Xtrain.shape[1]):
            corr = np.corrcoef(Xtrain[:,i], Ytrain,rowvar=False)
            corr = corr[0][1]
            if max_correlation <= abs(corr):
                max_correlation = abs(corr)
                index = i

        split_value = np.median(Xtrain[:,index])
        left_index=[]
        right_index=[]
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

        if Xtrain.shape[0] <= 1:
            return np.array([-1, np.mean(Ytrain), None, None])

        all_indexes = self.get_indexes(Xtrain, Ytrain)
        left_index, right_index, split_index, split_value = all_indexes[0], all_indexes[1], all_indexes[2], all_indexes[3]

        if len(left_index) == 0 or len(left_index) == Xtrain.shape[0]:
            return np.array([-1, np.mean(Ytrain), None, None])

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