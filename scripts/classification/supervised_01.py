import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()
print(digits['DESCR'][:193] + '\n')
print(f"Data shape: {digits.data.shape}")
print(f"Target shape: {digits.target.shape}")

# Display an example image
plt.imshow(digits.images[0], cmap='gray')
plt.title("Example Digit")
plt.axis('off')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)

# Create and train the classifier
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(gamma=0.001, C=100., random_state=42))
])
clf.fit(X_train, y_train)

# Predict using the last image in the test set
prediction = clf.predict(X_test[-1:])
plt.imshow(X_test[-1].reshape(8, 8), cmap='gray')
plt.title(f"Prediction: {prediction[0]}")
plt.axis('off')
plt.show()

# Evaluate the model
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

test_score = clf.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")