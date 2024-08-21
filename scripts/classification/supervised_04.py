import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Instantiate and train the decision tree classifier
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Example prediction
example = np.array([[5, 1.5, 3, 1.1]])
prob = clf.predict_proba(example)
prediction = clf.predict(example)
print("\nExample prediction:")
print(f"Input: {example[0]}")
print(f"Probabilities: {prob[0]}")
print(f"Predicted class: {iris.target_names[prediction[0]]}")

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True, 
               rounded=True, 
               fontsize=14)
plt.title("Decision Tree for Iris Dataset", fontsize=20)
plt.tight_layout()
plt.show()

# Feature importance
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(pos, feature_importance[sorted_idx], align='center')
ax.set_yticks(pos)
ax.set_yticklabels(np.array(iris.feature_names)[sorted_idx])
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance for Iris Classification')
plt.tight_layout()
plt.show()