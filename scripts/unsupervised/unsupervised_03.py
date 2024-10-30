import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
print(breast_cancer.DESCR)
print(breast_cancer.feature_names)


X = breast_cancer.data
y = breast_cancer.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Select the first two principal components for visualization
X_pca_2d = X_pca[:, :2]

# Create a scatter plot of the data points
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(scatter)
plt.show()

# Print the explained variance ratio of the first two components
print("Explained variance ratio of the first two components:")
print(pca.explained_variance_ratio_[:2])

# Print the feature names with the highest absolute coefficients for the first two PCs
feature_names = breast_cancer.feature_names
for i in range(2):
    pc = pca.components_[i]
    top_features = sorted(zip(feature_names, pc), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\nTop 5 features for PC{i+1}:")
    for name, coef in top_features:
        print(f"{name}: {coef:.4f}")