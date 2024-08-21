import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the digits dataset
digits = datasets.load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42
)

# Create a pipeline with StandardScaler and SVM classifier
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", C=5, random_state=42))
])

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate the classifier
accuracy = clf.score(X_test, y_test)
print(f"Mean accuracy on the test set: {accuracy:.4f}")

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="precision_macro")
print(f"5-fold cross-validation precision_macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Generate predictions for the test set
y_pred = clf.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))