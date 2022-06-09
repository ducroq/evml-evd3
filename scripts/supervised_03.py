import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset, see https://scikit-learn.org/stable/datasets/index.html#datasets
digits = datasets.load_digits()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, random_state=0)

# instantiate a classifier estimator
poly_kernel_svm_clf = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X_train, y_train)

### Set the parameters by cross-validation
##tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
##                     'C': [1, 10, 100, 1000]},
##                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
##
### Tuning hyper-parameters for accuracy
##clf = GridSearchCV(
##    SVC(), tuned_parameters, scoring='accuracy'
##)
##clf.fit(X_train, y_train)

##print("Grid scores on development set:")
##means = clf.cv_results_['mean_test_score']
##stds = clf.cv_results_['std_test_score']
##for mean, std, params in zip(means, stds, clf.cv_results_['params']):
##    print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))
##
##print("\nBest parameter on development set: {}".format(clf.best_params_))

score = cross_val_score(poly_kernel_svm_clf, X_train, y_train, cv=3, scoring="accuracy")
print("Accuracy: {}".format(score))

# Plotting decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots()#(2, 2, sharex='col', sharey='row', figsize=(10, 8))

Z = poly_kernel_svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axarr.contourf(xx, yy, Z, alpha=0.4)
axarr.scatter(X_train[:, 0], X_train[:, 1], c=y,
                              s=20, edgecolor='k')
axarr.set_title(tt)

plt.show()
