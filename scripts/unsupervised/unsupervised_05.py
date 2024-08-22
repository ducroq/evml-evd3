import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Create a DataFrame for easier handling
df = pd.DataFrame(X, columns=california.feature_names)
df['PRICE'] = y

# Function to plot feature distributions
def plot_feature_distributions(df, features, figsize=(15, 10)):
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        axes[i].hist(df[feature], bins=50)
        axes[i].set_title(feature)
        axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Plot original feature distributions
plot_feature_distributions(df, df.columns)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Isolation Forest
contamination = 0.1  # Assume 10% of the data are anomalies
clf = IsolationForest(contamination=contamination, random_state=42)
y_pred = clf.fit_predict(X_scaled)

# Add anomaly predictions to the DataFrame
df['ANOMALY'] = y_pred

# Separate normal and anomaly data
normal_data = df[df['ANOMALY'] == 1]
anomaly_data = df[df['ANOMALY'] == -1]

print(f"Number of normal data points: {len(normal_data)}")
print(f"Number of anomalies detected: {len(anomaly_data)}")

# Plot feature distributions with anomalies highlighted
def plot_feature_distributions_with_anomalies(normal_data, anomaly_data, features, figsize=(15, 10)):
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        axes[i].hist(normal_data[feature], bins=50, alpha=0.7, label='Normal')
        axes[i].hist(anomaly_data[feature], bins=50, alpha=0.7, label='Anomaly')
        axes[i].set_title(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Plot feature distributions with anomalies highlighted
plot_feature_distributions_with_anomalies(normal_data, anomaly_data, df.columns[:-2])

# Function to plot 2D scatter with anomalies highlighted
def plot_2d_scatter_with_anomalies(normal_data, anomaly_data, feature1, feature2):
    plt.figure(figsize=(10, 8))
    plt.scatter(normal_data[feature1], normal_data[feature2], c='blue', label='Normal', alpha=0.5)
    plt.scatter(anomaly_data[feature1], anomaly_data[feature2], c='red', label='Anomaly', alpha=0.5)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'{feature1} vs {feature2} with Anomalies')
    plt.legend()
    plt.show()

# Plot some interesting 2D relationships
plot_2d_scatter_with_anomalies(normal_data, anomaly_data, 'MedInc', 'PRICE')
plot_2d_scatter_with_anomalies(normal_data, anomaly_data, 'AveRooms', 'AveBedrms')

# Calculate feature importances
def get_feature_importances(clf, X):
    importances = []
    for i in range(X.shape[1]):
        X_temp = X.copy()
        X_temp[:, i] = np.random.permutation(X_temp[:, i])
        importances.append(np.mean(np.abs(clf.decision_function(X) - clf.decision_function(X_temp))))
    return np.array(importances)

# Get feature importances
importances = get_feature_importances(clf, X_scaled)

# Create a DataFrame of feature importances
feature_importance = pd.DataFrame({
    'feature': california.feature_names,
    'importance': importances
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nFeature Importances for Anomaly Detection:")
print(feature_importance)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importances for Anomaly Detection')
plt.show()