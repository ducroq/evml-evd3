import cv2 as cv
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# load dataset, see https://scikit-learn.org/stable/datasets/index.html#datasets
digits = datasets.load_digits()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

tuned_parameters = [{'n_neighbors': [3,5,7,9], 'weights': ['uniform', 'distance']}]

# Tuning hyper-parameters for accuracy
clf = GridSearchCV(
    KNeighborsClassifier(), tuned_parameters, scoring='accuracy'
)
clf.fit(X_train, y_train)

print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))

print("\nBest parameter on development set: {}".format(clf.best_params_))

score = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("Accuracy: {}".format(score))
