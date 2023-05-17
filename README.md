# Decision Tree Classifier

An implementation of Decision Tree classifier that recursively partitions data into smaller and smaller subsets until each subset is pure. It works by calculating impurity of
the root node and selecting the best feature and threshold to split the data at the node. The algorithm then creates two child nodes and recursively calls itself on each child node until all leaf nodes are pure.

Decision Tree relatively fast to train, and make it a good choice for tasks where speed is important. It is a good choice for tasks where it is important to be able to explain how the model works. While it is quite popular algorithm, some of the common reasons and behaviors to pay attention to when using Decision Tree:

- Notorious for overfitting the training data if they are allowed to grow too deep or if they are not regularized (pruning, penalty, boosting, ensemble) properly.
- Small variations in the training data can lead to instability (different shape of trees are generated)
- Less robust on continuous variable features
- Favors features with many categories (lead to poor generalization)


# Requirements

- Miniconda env
- Python 3.9
- Jupyter Notebook
- pytest

To install the required packages from the conda environment:

```
pip install -r requirements.txt
```

# Quick Start

To train a DecisionTree model, instantiates the `DecisionTreeClassifier` as follows and pass the features `X` tensor (samples, features) and target `y` (samples,) tensor:

```python
from decision_tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, y)
```

Below is an example of training a DecisionTree model on a sample animal data:

```python
import pandas as pd
from decision_tree import DecisionTreeClassifier

df = pd.read_csv('dataset/normalized_animals.csv')

# Split the DataFrame into features (X) and target (y).
X = df[['legs', 'color']].values
y = df['name'].values

clf = DecisionTreeClassifier()
clf.fit(X, y)

# Predict features of (legs:4, color:0).
cls = clf.predict([4,0])
print(cls)
```

Please look at the `decision_tree_playground.ipynb` for a working example how to train a DecisionTree and run prediction!

# Unittest

To run unittest:

```
pytest
```

To add more unittests, add to `decision_tree_test.py`.

# Dataset

There are two sample datasets in `dataset` folder. The `animals.csv` was the original dataset and the `normalized_animals.csv` is the dataset with the feature values transformed into integers. Current limitation of the library, it requires integer feature values for the current version.

# Future Works 

- Handle string feature values
- Add max depth regularization
- Add more unittests
- Test with larger dataset
- Add visualizer to describe the generated tree

