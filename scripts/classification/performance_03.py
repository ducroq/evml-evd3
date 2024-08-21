import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import os

def load_data():
    digits = datasets.load_digits()
    return train_test_split(digits.data, digits.target, test_size=0.4, random_state=42)

def create_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=0.5, random_state=42))
    ])

def plot_learning_curve(estimator, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.set_title("Learning Curves (SVM, Polynomial Kernel)")
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt

def save_plot(plt):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    filename = f"{script_name}_learning_curve.png"
    plt.savefig(filename)
    print(f"Learning curve plot saved as: {filename}")

def main():
    X_train, X_test, y_train, y_test = load_data()
    clf = create_classifier()
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    plt = plot_learning_curve(clf, X_train, y_train, cv=cv, n_jobs=-1)
    save_plot(plt)
    plt.show()

if __name__ == "__main__":
    main()