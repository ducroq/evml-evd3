import cv2 as cv
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# load dataset, see https://scikit-learn.org/stable/datasets/index.html#datasets
digits = datasets.load_digits()
##iris = datasets.load_iris()

# the .data member, which is a (n_samples, n_features) sized array.
# labels are  stored in the .target member
print("feature array: {}".format(digits.data))
print("label array: {}".format(digits.target))

# the original image data can also be retreived
img = digits.images[0]
cv.imshow("Example image", img)

# instantiate a classifier estimator
clf = Pipeline([
("svm_clf", svm.SVC(gamma=0.001, C=100.))
])

# train the model on all images except the last 
clf.fit(digits.data[:-1], digits.target[:-1])

# predict using the last image
prediction = clf.predict(digits.data[-1:])
cv.imshow("Prediction, digit = {:}".format(prediction[0]), digits.images[-1])

# compute accuracy over n-folds
X_train, X_test = digits.data[:1000,], digits.data[1000:,]
y_train, y_test = digits.target[:1000,], digits.target[1000:,]
score = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("score: {}".format(score))
