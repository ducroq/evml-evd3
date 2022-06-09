import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


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
X = diabetes.data[:, 2, np.newaxis]
# note that a 2D array is expected by sklearn functions, hence a new axis

# labels
y = diabetes.target


# Split the dataset in two parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Create linear regression object
degree = 3
reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

# Train the model using the training sets
reg.fit(X_train, y_train)

# Make predictions 
y_pred = reg.predict(X_test)


### Results
print('Regression coefficients: {}'.format(reg['lin_reg'].coef_))

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(X_test[:,0], y_test,  color='black')
X_plot = np.array(np.arange(-.1,.2,.01), ndmin=2).transpose()
y_plot = reg.predict(X_plot)

ax.plot(X_plot[:,0], y_plot, color='blue', linewidth=3)
ax.set_xlabel('normalized ' + diabetes.feature_names[2])
ax.set_ylabel('disease progression')
ax.legend(['prediction'])
ax.set_title('Degree {} polynomial regression'.format(degree))
ax.axis([min(X),max(X),min(y_pred),max(y_pred)])
plt.show(block=True)
