import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in"""
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier."""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Load and prepare the dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features for visualization
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
C = 1.0  # SVM regularization parameter
models = (
    ('SVC with linear kernel', svm.SVC(kernel='linear', C=C)),
    ('LinearSVC (linear kernel)', svm.LinearSVC(C=C, max_iter=10000)),
    ('SVC with RBF kernel', svm.SVC(kernel='rbf', gamma=0.7, C=C)),
    ('SVC with polynomial (degree 3) kernel', svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
)

# Set up the plot
fig, sub = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf_name, clf in models:
    # Train the model
    clf.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot decision boundary
    ax = sub.flatten()[models.index((clf_name, clf))]
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f'{clf_name}\nAccuracy: {accuracy:.2f}')

plt.suptitle("SVM Decision Boundaries on Iris Dataset", fontsize=16)
plt.tight_layout()
plt.show()