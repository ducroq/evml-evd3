import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
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


# Load dataset
iris = datasets.load_iris()

# print some info
# refer to https://docs.python.org/3/library/string.html#formatspec
# to learn about the Format Specification Mini-Language
print(iris.DESCR)
print("Attributes:{}".format(dir(iris)))
print("Features: {}".format(iris.feature_names))
print("Nr of samples: {:d}".format(iris.target.size))


# Select features
feature_nr = 3
X = iris.data[:, feature_nr, np.newaxis]

# labels
##y = iris.target
y = (iris["target"] == 2).astype(int) # 1 if Iris virginica, else 0


# Split the dataset in two  parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Create regression object
degree = 2
reg = Pipeline([    
    ("log_reg", LogisticRegression()),
])


reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(X_test, y_test,  color='black')
X_plot = np.linspace(0, 3, 1000).reshape(-1, 1)
y_plot = reg.predict(X_plot)

ax.plot(X_plot[:,0], y_plot, color='blue', linewidth=3)
##ax.plot(X_test, y_pred, color='blue', linewidth=3)
ax.set_xlabel(iris.feature_names[feature_nr])
ax.set_ylabel('probability')
ax.legend(['prediction']) 


plt.show(block=True)
