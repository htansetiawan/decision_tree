import numpy as np
import pandas as pd
from decision_tree import DecisionTreeClassifier


def prepare_data():
    array = np.array([
        [4, 0, 0],
        [4, 2, 1],
        [2, 1, 2],
        [2, 2, 3],
        [0, 2, 4],
        [3, 1, 5],
        [3, 2, 2],
    ])

    column_values = ['neurons', 'eyes', 'class']

    df = pd.DataFrame(data = array,
                      columns = column_values)
    return df


def compute_gini_reference(classes):
    """Compute gini impurity reference."""
    _, counts = np.unique(classes, return_counts=True)
    probabilities = counts / len(classes)
    gini_impurity = 1.0 - sum(probabilities**2)
    return gini_impurity


def test_create():
    clf = DecisionTreeClassifier()
    assert clf is not None
    assert clf.tree is not None


def test_predict():
    clf = DecisionTreeClassifier()
    df = prepare_data()
    X = df[['neurons', 'eyes']].values
    y = df['class'].values
    clf.fit(X, y)
    assert 0 == clf.predict([4, 0])
    assert 1 == clf.predict([4, 2])
    assert 2 == clf.predict([2, 1])
    assert 3 == clf.predict([2, 2])
    assert 4 == clf.predict([0, 2])
    assert 5 == clf.predict([3, 1])


def test_compute_gini():
    clf = DecisionTreeClassifier()
    df = prepare_data()
    classes = df['class'].values
    assert compute_gini_reference(classes) == clf._compute_gini(classes)
    classes = df['eyes'].values
    assert compute_gini_reference(classes) == clf._compute_gini(classes)
    classes = df['neurons'].values
    assert compute_gini_reference(classes) == clf._compute_gini(classes)


def test_split_data():
    """Test splitting the X, y by the feature 0, with threshold 2.
    X: [[4 0]
       [4 2]
       [2 1]
       [2 2]
       [0 2]
       [4 1]]
    y: [0 1 2 3 4 5]

    X_left: [[0 2]]
    y_left: [4]

    X_right: [[4 0]
              [4 2]
              [2 1]
              [2 2]
              [4 1]]

    Y_right: [0 1 2 3 5]
    """
    clf = DecisionTreeClassifier()
    df = prepare_data()
    X = df[['neurons', 'eyes']].values
    y = df['class'].values
    X_left, y_left, X_right, y_right = clf._split_data(X, y, 0, 2)
    assert (1, 2) == X_left.shape
    assert np.array_equal([[0, 2]], X_left)
    assert (1,) == y_left.shape
    assert np.array_equal([4], y_left)
    assert (6, 2) == X_right.shape
    assert (6,) == y_right.shape
    assert np.array_equal([[4, 0],[4, 2],[2, 1],[2, 2],[3, 1], [3, 2]], X_right)
    assert np.array_equal([0, 1, 2, 3, 5, 2], y_right)


def test_best_split():
  """
  Manual computation:
  gini:0.6666666666666666 feature_idx:0 threshold:2
  gini:0.7142857142857143 feature_idx:0 threshold:3
  gini:0.657142857142857 feature_idx:0 threshold:4
  gini:0.6666666666666665 feature_idx:1 threshold:1
  gini:0.7142857142857143 feature_idx:1 threshold:2
  """
  clf = DecisionTreeClassifier()
  df = prepare_data()
  X = df[['neurons', 'eyes']].values
  y = df['class'].values
  # based on computation by hand.
  assert (0, 4) == clf._do_best_split(X, y)
