import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Load a subset of the digits dataset for faster processing
digits = datasets.load_digits()
subset_size = 1000  # Adjust this value based on your performance needs
subset_indices = np.random.choice(len(digits.data), subset_size, replace=False)
X_subset = digits.data[subset_indices]
y_subset = digits.target[subset_indices]

# Choose two features for visualization (change these as desired)
feature_1 = 1  # Index of the first feature to visualize
feature_2 = 4  # Index of the second feature to visualize

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.5, random_state=0)

# Create a pipeline with StandardScaler and SVM classifier
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

# Fit the classifier
poly_kernel_svm_clf.fit(X_train, y_train)

# Perform cross-validation
scores = cross_val_score(poly_kernel_svm_clf, X_train, y_train, cv=3, scoring="accuracy")
print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Visualize decision boundary for the selected features
plt.figure(figsize=(10, 8))

# Create a mesh to plot in
x_min, x_max = X_train[:, feature_1].min() - 1, X_train[:, feature_1].max() + 1
y_min, y_max = X_train[:, feature_2].min() - 1, X_train[:, feature_2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

# Create full feature vectors for prediction
XX = np.c_[xx.ravel(), yy.ravel()]
XX_full = np.zeros((XX.shape[0], X_train.shape[1]))
XX_full[:, feature_1] = XX[:, 0]
XX_full[:, feature_2] = XX[:, 1]

# Predict
Z = poly_kernel_svm_clf.predict(XX_full)
Z = Z.reshape(xx.shape)

# Plot the contour
plt.contourf(xx, yy, Z, alpha=0.4)
scatter = plt.scatter(X_train[:, feature_1], X_train[:, feature_2], c=y_train, s=20, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(),
                     loc="center left", bbox_to_anchor=(1, 0.5),
                     title="Digits")
plt.gca().add_artist(legend1)

plt.title(f'SVM with Polynomial Kernel Decision Regions\n(Features {feature_1} and {feature_2})')
plt.xlabel(f'Feature {feature_1}')
plt.ylabel(f'Feature {feature_2}')

plt.show()