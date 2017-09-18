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
        split_index = -1;
        max_correlation = 0
        for i in range(Xtrain.shape[0]):
            corr = np.corrcoef(Xtrain[:,i], Ytrain)
            corelation.append(corr)
            if max_correlation <= abs(corr):
                max_correlation = abs(corr)
                split_index = i

        split_value = Xtrain[:,i].median()
        left_index = []
        right_index = []
        for i in range(Xtrain.shape[0]):
            if Xtrain[i][split_index] <= split_value:
                left_index.append(i)
            else:
                right_index.append(i)

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

    def traverse_tree(self, instance, row=0):
        feature_index = int(self.tree[row][0])
        if feature_index == -1:
            return self.tree[row][1]
        if instance[feature_index] <= self.tree[row][1]:
            return self.traverse_tree(instance, row + int(self.tree[row][2]))
        else:
            return self.traverse_tree(instance, row + int(self.tree[row][3]))

    def query(self, Xtest):
        result = []
        for instance in Xtest:
            result.append(self.traverse_tree(instance))
        return np.array(result)

