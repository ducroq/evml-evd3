# Applies t-SNE to reduce the dimensionality of the data to 2D
# Creates a scatter plot of the t-SNE results, with each point colored according to its digit class.
# Uses a K-Nearest Neighbors classifier to evaluate how well t-SNE has separated the classes.

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_train_scaled)

# Visualize the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of the Digits dataset')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()

# Function to plot sample digits
def plot_digits(X, y, num_samples=25):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(X[i].reshape(8, 8), cmap='gray')
            ax.set_title(f'Digit: {y[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plot sample digits
plot_digits(X_train, y_train)

# Evaluate the separation using a simple classifier (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tsne, y_train)

# Make predictions on the t-SNE transformed data
y_pred = knn.predict(X_tsne)

# Calculate and print the accuracy
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy of KNN on t-SNE transformed data: {accuracy:.2f}")

# Compare with accuracy on original data
knn_original = KNeighborsClassifier(n_neighbors=5)
knn_original.fit(X_train_scaled, y_train)
y_pred_original = knn_original.predict(X_test_scaled)
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"Accuracy of KNN on original data: {accuracy_original:.2f}")

# Calculate and print silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_tsne, y_train)
print(f"The average silhouette score is: {silhouette_avg:.2f}")