import cv2 as cv
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load dataset, see https://scikit-learn.org/stable/datasets/index.html#datasets
iris = datasets.load_iris()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1)

# instantiate a classifier estimator
clf = tree.DecisionTreeClassifier(max_depth=2)

# train the model
clf.fit(X_train, y_train)

# check out class probability
print(clf.predict_proba([[5, 1.5, 3, 1.1]]))
print(clf.predict([[5, 1.5, 3, 1.1]]))

# show tree
plt.figure()
tree.plot_tree(clf, feature_names=iris.feature_names,
               class_names=iris.target_names,
               rounded=True,
               filled=True)
plt.show()

