import cv2 as cv
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load dataset, see https://scikit-learn.org/stable/datasets/index.html#datasets
digits = datasets.load_digits()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, random_state=0)

# instantiate a classifier estimator
clf = tree.DecisionTreeClassifier(max_depth=10)

# train the model
clf.fit(X_train, y_train)

# show tree
plt.figure()
tree.plot_tree(clf, feature_names=digits.feature_names,
               class_names=[str(i) for i in digits.target_names],
               rounded=True,
               filled=True)
plt.show()

