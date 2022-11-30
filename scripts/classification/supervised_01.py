import cv2
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

# load dataset, see https://scikit-learn.org and find the digit dataset
digits = datasets.load_digits()
# iris = datasets.load_iris()
print(digits['DESCR'][:193] + '\n')

# the .data member, which is a (n_samples, n_features) sized array.
# labels are  stored in the .target member
print("feature array: {}".format(digits.data))
print("label array: {}".format(digits.target))

# the original image data can also be retreived
img = digits.images[0]
cv2.imshow("Example image", img)
cv2.waitKey(0)

# instantiate a classifier estimator
# clf = svm.SVC(gamma=0.001, C=100.)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = Pipeline([('scaler', StandardScaler()),
                ('classifier', svm.SVC(gamma=0.001, C=100.))])

# train the model on all images except the last 
clf.fit(digits.data[:-1], digits.target[:-1])

# predict using the last image
prediction = clf.predict(digits.data[-1:])
cv2.imshow("Prediction, digit = {:}".format(prediction[0]), digits.images[-1])
cv2.waitKey(0)



# compute accuracy over n-folds
X_train, X_test = digits.data[:1000,], digits.data[1000:,]
y_train, y_test = digits.target[:1000,], digits.target[1000:,]
score = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print("score: {}".format(score))


print(clf.score(X_test, y_test))
