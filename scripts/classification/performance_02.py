import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load the digits dataset
digits = datasets.load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.1, random_state=42
)

# Create a pipeline with StandardScaler and SVM classifier
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="linear", C=5, random_state=42))
])

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix for Digits Classification')
plt.tight_layout()

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Show the plot
plt.show()

# Optionally, save the confusion matrix plot
# import os

# script_name = os.path.splitext(os.path.basename(__file__))[0]
# confusion_matrix_filename = f"{script_name}_confusion_matrix.png"
# plt.savefig(confusion_matrix_filename)
# print(f"Confusion matrix saved as: {confusion_matrix_filename}")
