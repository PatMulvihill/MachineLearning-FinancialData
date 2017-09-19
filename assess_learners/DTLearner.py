import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size, verbose):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return 'lwang496'

    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

    def get_indexes(self, Xtrain, Ytrain):
        corelation = []
        index = -1;
        max_correlation = 0
        for i in range(Xtrain.shape[0]):
            corr = np.corrcoef(Xtrain[:,i], Ytrain)
            corelation.append(corr)
            if max_correlation <= abs(corr):
                max_correlation = abs(corr)
                index = i

        split_value = Xtrain[:,index].median()
        left_index=[]
        right_index=[]
        for i in xrange(Xtrain.shape[0]):
            if Xtrain[i][index] <= split_value:
                left_index.append(i)
            else:
                right_index.append(i)

        return left_index, right_index, index, split_value

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
            left_index, right_index, split_index, split_value = self.get_indexes(Xtrain)

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

