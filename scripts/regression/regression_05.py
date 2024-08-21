# Wine Quality Dataset Polynomial Regression
# This script demonstrates polynomial regression using the Wine Quality dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import io
import zipfile
import requests

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
csv_file = zip_file.open("winequality-red.csv")

# Read the CSV file
df = pd.read_csv(csv_file, delimiter=';')

# We'll use 'alcohol' content to predict 'quality'
X = df['alcohol'].values.reshape(-1, 1)
y = df['quality'].values

# Note that features like 'volatile acidity' and 'sulphates' often show non-linear relationships with quality

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models with different polynomial degrees
degrees = [1, 2, 3]
colors = ['r', 'g', 'b']
plt.figure(figsize=(12, 8))

for degree, color in zip(degrees, colors):
    # Create and train the polynomial regression model
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print the results
    print(f"\nPolynomial Degree {degree}:")
    print(f"Mean squared error: {mse:.4f}")
    print(f"R-squared score: {r2:.4f}")
    
    # Plot the results
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Actual' if degree == 1 else '')
    plt.plot(X_plot, y_plot, color=color, label=f'Degree {degree}')

plt.xlabel('Alcohol Content')
plt.ylabel('Wine Quality')
plt.title('Wine Quality vs. Alcohol Content (Polynomial Regression)')
plt.legend()
plt.show()

# Optional: Print coefficients for the highest degree model
highest_degree_model = model.named_steps['linearregression']
coeffs = highest_degree_model.coef_
intercept = highest_degree_model.intercept_
print(f"\nCoefficients for degree {degrees[-1]} polynomial:")
for i, coef in enumerate(coeffs):
    print(f"  Degree {i}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")