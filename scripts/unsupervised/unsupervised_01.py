# K-Means Clustering on Iris Dataset
# This script demonstrates K-Means clustering using the Iris dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
n_clusters = 3  # We know there are 3 species in the Iris dataset
kmeans = KMeans(n_clusters=n_clusters)#, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Define a discrete color map
colors = ['#FF9999', '#66B2FF', '#99FF99']
cmap = plt.cm.colors.ListedColormap(colors)

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot 1: Sepal length vs Sepal width
plt.subplot(121)
scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap=cmap)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means Clustering on Iris Dataset\nSepal Features')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Cluster')

# Plot 2: Petal length vs Petal width
plt.subplot(122)
scatter = plt.scatter(X[:, 2], X[:, 3], c=cluster_labels, cmap=cmap)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('K-Means Clustering on Iris Dataset\nPetal Features')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Cluster')

plt.tight_layout()
plt.show()

# Print cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
feature_names = iris.feature_names
print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}:")
    for j, value in enumerate(center):
        print(f"  {feature_names[j]}: {value:.2f}")

# Evaluate the clustering
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# Compare with true labels
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y, cluster_labels)
print(f"Adjusted Rand Index: {ari:.3f}")

