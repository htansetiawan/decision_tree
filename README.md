# Decision Tree Classifier

An implementation of Decision Tree classifier that recursively partitions data into smaller and smaller subsets until each subset is pure. It works by calculating impurity of
the root node and selecting the best feature and threshold to split the data at the node. The algorithm then creates two child nodes and recursively calls itself on each child node until all leaf nodes are pure.

# Requirements

- Miniconda env
- Python 3.9

To install the required packages from the conda environment:

```
pip install -r requirements.txt
```

# Quick Start

Usage example:

```python
import pandas as pd
from decision_tree import DecisionTreeClassifier

df = pd.read_csv('dataset/animals_clean.csv')

# Split the DataFrame into features (X) and target (y).
X = df[['legs', 'color']].values
y = df['name'].values

clf = DecisionTreeClassifier()
clf.fit(X, y)

# Predict features of (legs:4, color:0).
predictions = clf.predict([4,0])
print(predictions)
```

Please look at the `decision_tree_playground.ipynb` for a working example how to train a DecisionTree and run prediction!

# Dataset

There are two datasets in `dataset` folder. The `animals.csv` was the original dataset and the `animals_clean.csv` is the dataset with values transformed into integers.
The implementation of the library requires integer feature values for the current version.

As a future work, the library could handle the string features value.
