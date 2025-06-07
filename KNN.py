import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from main import data_loader

N_NEIGHBORS = [99, 99, 95, 95, 97, 97]

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--optimize", action="store_true", help="Optimize the number of neighbors (k) for KNN")
parser.add_argument(
    "-w",
    "--which",
    type=int,
    default=3,
    help="Choose the unit of kbar, 1 means 15mins, 3 means 1hr, and 5 means 4hr. 2, 4, and 6 means find the difference between each two kbar",
)
args = parser.parse_args()


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: precision, recall, and F1-score
    """
    # Get all true positives, false positives, and false negatives from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate global metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    cm_display = np.array([[tp, fn], [fp, tn]])
    print(f"Confusion Matrix:\n{cm_display}")
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Positive", "Negative"])
    plt.yticks(tick_marks, ["Positive", "Negative"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations in the cells
    thresh = (tp + tn) / 2
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm_display[i][j],
                horizontalalignment="center",
                color="white" if cm_display[i][j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    return accuracy, precision, recall, f1_score


def find_optimal_k(X, y, k_range=range(1, 101, 2)):
    """
    Use cross-validation to find the optimal number of neighbors for KNN

    Parameters:
    -----------
    X : array-like
        Training data
    y : array-like
        Target values
    k_range : range or list
        The range of k values to test

    Returns:
    --------
    best_k : int
        The optimal k value
    cv_scores : dict
        Dictionary containing k values and their corresponding cross-validation scores
    """
    cv_scores = {}

    for k in k_range:
        print(f"Testing k = {k}...")
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring="f1")
        cv_scores[k] = scores.mean()

    # Find the best k value
    best_k = max(cv_scores, key=cv_scores.get)

    # Plot the cross-validation results
    plt.figure(figsize=(10, 6))
    plt.plot(list(cv_scores.keys()), list(cv_scores.values()), marker="o")
    plt.axvline(x=best_k, color="r", linestyle="--", label=f"Best k = {best_k}")
    plt.title("Cross-Validation Scores for Different K Values")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("F1 Score (Cross-Validation)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"optimal_k.png", dpi=300, bbox_inches="tight")

    print(f"The optimal number of neighbors is: {best_k} with F1 score: {cv_scores[best_k]:.4f}")


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = data_loader(which=args.which)
    print(f"========== Data Loaded for unit {args.which} ==========")
    train_data = train_data.T
    test_data = test_data.T

    if args.optimize:
        print("Optimizing the number of neighbors (k) for KNN...")
        find_optimal_k(train_data, train_label)
    else:
        print(f"Using KNN with {N_NEIGHBORS[args.which - 1]} neighbors...")
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS[args.which - 1])
        knn.fit(train_data, train_label)
        y_pred = knn.predict(test_data)
        accuracy, precision, recall, f1_score = calculate_evaluation_metrics(test_label, y_pred)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
