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

    def get_split_indices(self, Xtrain, Ytrain):
        corelation = []
        split_index = -1;
        max_correlation = 0
        for i in range(Xtrain.shape[0]):
            corr = np.corrcoef(Xtrain[:,i], Ytrain)
            corelation.append(corr)
            if max_correlation <= abs(corr):
                max_correlation = abs(corr)
                split_index = i

        split_value = Xtrain[:,split_index].median()
        left_index = [i for i in xrange(Xtrain.shape[0])
                        if Xtrain[i][split_index] <= split_value]
        right_index = [i for i in xrange(Xtrain.shape[0])
                         if Xtrain[i][split_index] > split_value]

        return left_index, right_index, split_index, split_value

    def build_tree(self, x_train, y_train):

        num_instances = x_train.shape[0]
        if num_instances == 0:
            print 'all -1s'
            return np.array([-1, -1, -1, -1])
        if num_instances <= self.leaf_size:
            # If there's only one instance, take the mean of the labels
            return np.array([-1, np.mean(y_train), -1, -1])

        values = np.unique(y_train)
        if len(values) == 1:
            # If all instances have the same label, return that label
            return np.array([-1, y_train[0], -1, -1])

        # Choose a random feature, and a random split value
        left_indices, right_indices, feature_index, split_val = \
            self.get_split_indices(x_train, num_instances)



        left_x_train = np.array([x_train[i] for i in left_indices])
        left_y_train = np.array([y_train[i] for i in left_indices])
        right_x_train = np.array([x_train[i] for i in right_indices])
        right_y_train = np.array([y_train[i] for i in right_indices])

        left_tree = self.build_tree(left_x_train, left_y_train)
        right_tree = self.build_tree(right_x_train, right_y_train)
        if len(left_tree.shape) == 1:
            num_left_side_instances = 2
        else:
            num_left_side_instances = left_tree.shape[0] + 1
        root = [feature_index, split_val, 1, num_left_side_instances]
        return np.vstack((root, np.vstack((left_tree, right_tree))))

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

