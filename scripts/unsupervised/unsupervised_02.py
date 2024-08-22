import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering
n_clusters = 3  # We know there are 3 classes in the Wine dataset
clustering = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True)
clustering.fit(X_scaled)

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plot_dendrogram(clustering, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()

# Perform PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clustering.labels_, cmap='viridis')
plt.title('Hierarchical Clustering of Wine Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter)
plt.show()

# Print cluster sizes
unique, counts = np.unique(clustering.labels_, return_counts=True)
print("\nCluster sizes:")
for i, count in zip(unique, counts):
    print(f"Cluster {i}: {count} samples")

# Compare with true labels
print("\nClustering performance:")
print(f"Adjusted Rand Index: {adjusted_rand_score(y, clustering.labels_):.2f}")
print(f"Adjusted Mutual Information: {adjusted_mutual_info_score(y, clustering.labels_):.2f}")

# Find the most distinguishing features for each cluster
feature_names = wine.feature_names
cluster_centers = np.array([X_scaled[clustering.labels_ == i].mean(axis=0) for i in range(n_clusters)])

print("\nMost distinguishing features for each cluster:")
for i in range(n_clusters):
    sorted_features = sorted(zip(feature_names, cluster_centers[i]), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nCluster {i}:")
    for name, value in sorted_features[:3]:
        print(f"{name}: {value:.2f}")