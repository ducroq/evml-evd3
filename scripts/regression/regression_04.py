import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import io
import zipfile
import requests

# Download and load the Air Quality dataset
url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
csv_file = zip_file.open("AirQualityUCI.csv")

# Read the CSV file
df = pd.read_csv(csv_file, sep=';', decimal=',')

# Clean and prepare the data
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop unnamed columns
df.replace(-200, np.nan, inplace=True)  # Replace -200 with NaN as it denotes missing values
df.dropna(inplace=True)  # Drop rows with NaN
# print(df.head)

# Select feature and target
feature = 'T'
target = 'C6H6(GT)'

X = df[[feature]]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate and print the model's performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean squared error: {mse:.4f}")
print(f"R-squared score: {r2:.4f}")

# Print the coefficient and intercept
print("\nModel Coefficient and Intercept:")
print(f"{feature} coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Visualize the results with regression line
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred, color='blue', alpha=0.5, label='Predicted')

# Regression line
x_range = np.linspace(X_test[feature].min(), X_test[feature].max(), 100).reshape(-1, 1)
y_range = model.predict(scaler.transform(x_range))
plt.plot(x_range, y_range, color='red', label='Regression Line')

plt.xlabel(feature)
plt.ylabel(target)
plt.title(f'Linear regression of {feature} vs {target}')
plt.legend()

# Add text annotations for model performance
plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}', transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
