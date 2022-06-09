import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error #plot_confusion_matrix, classification_report


### plots of the model’s performance
## on the training set and the validation set as a function of the training set
## size (or the training iteration). To generate the plots, train the model several times on
## different sized subsets of the training set.
def learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    
    for m in range(2, len(X_train)):
        print("Training and validating completion: {:.1}%".format(100*m/len(X_train)), end='\r')
        model.fit(X_train[:m], y_train[:m])
        
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    print()
        
    return train_errors, val_errors


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
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])

# fit the classifier
clf.fit(X_train, y_train)

### Plot confusion matrix
##fig0, ax0 = plt.subplots(1,1)
##plot_confusion_matrix(clf, X_test, y_test, ax=ax0)
##ax0.set_title('Confusion matrix')
##plt.tight_layout()
##
### Show detailed classification report
##y_true, y_pred = y_test, clf.predict(X_test)
##print("Detailed classification report:")
##print()
##print(classification_report(y_true, y_pred))
##print()

train_errors, val_errors = learning_curves(clf, X_train, y_train)


# Plot outputs
fig, ax = plt.subplots()
ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
ax.set_title('Classification learning curves')
ax.set_xlabel('training set size')
ax.set_ylabel('RMSE')
ax.axis([0,len(train_errors),0,10])
ax.legend(['train', 'val'])

plt.show(block=False)




