import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
feature_nr = 2
X = diabetes.data[:, feature_nr, np.newaxis]
# note that a 2D array is expected by sklearn functions, hence a new axis

# labels
y = diabetes.target

# Split the dataset in two  parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)


# Results
print('Regression line: y = {:.2} + {:.2} x'.format(reg.intercept_, reg.coef_[0]))

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(X_test, y_test,  color='black')
ax.plot(X_test, y_pred, color='blue', linewidth=3)
ax.set_xlabel('normalized ' + diabetes.feature_names[feature_nr])
ax.set_ylabel('disease progression')
ax.legend(['predicition']) 
ax.set_title('diabetes progression') 

plt.show(block=True)
