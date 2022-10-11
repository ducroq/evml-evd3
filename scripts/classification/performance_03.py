import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
from sklearn.preprocessing import StandardScaler


# load a dataset, see https://scikit-learn.org/stable/datasets/index.html#datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Sample a training set while holding out 40% of the data for testing (evaluating) our classifier
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.4, random_state=0)

# instantiate a classifier estimator
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=1, coef0=100, C=0.5))
    ])

# Propose cross-validation indices in the data set
cv = ShuffleSplit(n_splits=100, test_size=0.2)#, random_state=0)

# Compute learning curves
train_sizes=np.linspace(.1, 1.0, 10)
train_sizes, train_scores, test_scores = \
    learning_curve(clf, X_train, y_train, cv=cv, n_jobs=-1,
                   train_sizes=train_sizes)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)



# Plot learning curves
fig, ax = plt.subplots(1, 1)
ax.grid()
ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
ax.set_title('Learning Curves for SVM, poly kernel, degree = 3')
ax.set_xlabel('training set size')
ax.set_ylabel('Score')
ax.legend(['train', 'val'],loc='lower right')

plt.show(block=True)




