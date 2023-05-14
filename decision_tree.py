import numpy as np

class DecisionTreeClassifier:
    """Decision tree classifier.

    Recursively partitions data into smaller and smaller subsets
    until each subset is pure. It works by calculating impurity of
    the root node and selecting the best feature and threshold to
    split the data at the node. The algorithm then creates two child
    nodes and recursively calls itself on each child node until
    all leaf nodes are pure.
    """

    def __init__(self):
        self.tree = {}

    def fit(self, features, targets):
        """Trains a decision tree.

        Args:
            features: the feature matrix of shape (n_samples, n_features).
            targets: the target vector of shape (n_samples,).
        """
        self.tree = self._build_tree(features, targets)

    def predict(self, features):
        """Predicts the class label for a given set of features.

        Args:
            features: the features to predict on.

        Returns:
            The predicted class.
        """
        node = self.tree
        # Traverse the decision tree until we find a leaf node.
        while not node['leaf']:
            if features[node['feature_idx']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['label']

    def _build_tree(self, features, targets):
        """Recursively builds the decision tree.

        Args:
            features: the feature matrix of shape (n_samples, n_features).
            targets: the target vector of shape (n_samples,).

        Returns:
            The root node.
        """
        sample_count, feature_count = features.shape
        label_count = len(set(targets))

        # Terminal conditions.
        if sample_count == 0:
            return {'label': None, 'leaf': True}
        if label_count == 1:
            return {'label': targets[0], 'leaf': True}

        split_feature_idx, split_threshold = self._do_best_split(features, targets)
        if split_feature_idx is None:
            return {'label': None, 'leaf': True}
        X_left, y_left, X_right, y_right = self._split_data(features, targets, split_feature_idx, split_threshold)

        # Recursive calls
        left_subtree = self._build_tree(X_left, y_left)
        right_subtree = self._build_tree(X_right, y_right)
        return {
            'feature_idx': split_feature_idx,
            'threshold': split_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'leaf': False
        }


    def _do_best_split(self, features, targets):
        """Finds the best features and threshold to split the data based on the gini impurity.

        Args:
            features: the feature matrix of shape (n_samples, n_features).
            targets: the target vector of shape (n_samples,).

        Returns:
            A tuple of (best feature index, threshold).
        """
        sample_count, feature_count = features.shape
        split_feature_idx, split_threshold, split_gini = None, None, 1.0

        for feature_idx in range(feature_count):
            feature_values = features[:, feature_idx]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(features, targets, feature_idx, threshold)
                if len(X_left) == 0 or len(X_right) == 0:
                    continue
                gini_left = self._compute_gini(y_left)
                gini_right = self._compute_gini(y_right)
                weighted_gini = (len(X_left) / sample_count) * gini_left + (len(X_right) / sample_count) * gini_right
                if weighted_gini < split_gini:
                    split_feature_idx = feature_idx
                    split_threshold = threshold
                    split_gini = weighted_gini

        return split_feature_idx, split_threshold


    def _split_data(self, features, targets, feature_idx, threshold):
        """Splits the data into left and right subsets based on a given feature and threshold.

        Args:
            features: the feature matrix of shape (n_samples, n_features).
            targets: the target vector of shape (n_samples,).
            feature_idx: the index of the feature to split on.
            threshold: the threshold value to split on.

        Returns:
            A tuple of (left-feature, left-target, right-feature, right-vector).
        """
        idx_left = features[:, feature_idx] < threshold
        idx_right = features[:, feature_idx] >= threshold
        X_left, y_left = features[idx_left], targets[idx_left]
        X_right, y_right = features[idx_right], targets[idx_right]
        return X_left, y_left, X_right, y_right


    def _compute_gini(self, classes):
        """Computes gini impurity.

        Measures how often a randomly chosen element of a set would be
        incorrectly labeled if it was labeled randomly and independently
        according to the distribution of labels in the set. It reaches its
        minimum (zero) when all cases in the node fall into a single target category.

        source:https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity.

        Args:
            classes: the target vector of shape (n_samples,).
        """
        _, counts = np.unique(classes, return_counts=True)
        probabilities = counts / len(classes)
        gini_impurity = 1.0 - sum(probabilities**2)
        return gini_impurity

