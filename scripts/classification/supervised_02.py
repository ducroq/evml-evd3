import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC  # Uncomment if using SVM
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

# Alternative parameter grid for SVM (commented out)
# param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                'C': [1, 10, 100, 1000]},
#               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Create and run the GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
# If using SVM, replace the line above with:
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Print grid scores
print("\nGrid scores on development set:")
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print(f"{mean:.3f} (+/-{std * 2:.03f}) for {params}")

# Evaluate on the test set
test_score = grid_search.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.3f}")

# Visualize the results (for KNN only)
if 'n_neighbors' in param_grid:
    n_neighbors = param_grid['n_neighbors']
    train_scores = grid_search.cv_results_['mean_test_score'].reshape(len(n_neighbors), -1)

    plt.figure(figsize=(10, 6))
    plt.plot(n_neighbors, train_scores[:, 0], label='uniform weights')
    plt.plot(n_neighbors, train_scores[:, 1], label='distance weights')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN Performance by Number of Neighbors and Weight Type')
    plt.legend()
    plt.show()

# Display an example digit
plt.figure(figsize=(5, 5))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Example: Digit 0")
plt.axis('off')
plt.show()