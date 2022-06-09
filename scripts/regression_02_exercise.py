import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

### plots of the model’s performance
## on the training set and the validation set as a function of the training set
## size (or the training iteration). To generate the plots, train the model several times on
## different sized subsets of the training set.

def learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        
    return train_errors, val_errors


# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# print some info
# refer to https://docs.python.org/3/library/string.html#formatspec
# to learn about the Format Specification Mini-Language
print(diabetes.DESCR)
print("Attributes:{}".format(dir(diabetes)))
print("Features: {}".format(diabetes.feature_names))
print("Nr of samples: {:d}".format(diabetes.target.size))


# Use only BMI as a feature
# note that variables have been normalized to have mean 0 and squared length = 1 
X = diabetes.data[:,  2, np.newaxis] #0:5] #
# note that a 2D array is expected by sklearn functions, hence a new axis

# labels
y = diabetes.target

degree = 2
reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

train_errors, val_errors = learning_curves(reg, X, y)


# Plot outputs
fig, ax = plt.subplots()
ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
ax.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
ax.set_title('Learning curves for degree {} polynomial regression'.format(degree))
ax.set_xlabel('training set size')
ax.set_ylabel('RMSE')
ax.axis([0,len(train_errors),0,100])
ax.legend(['train', 'val'])

plt.show(block=True)
