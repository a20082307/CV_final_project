from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# import umap
# from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from main import data_loader


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: precision, recall, and F1-score
    """
    # Get all true positives, false positives, and false negatives from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate global metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    cm_display = np.array([[tp, fp], [fn, tn]])
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
    plt.show()

    return precision, recall, f1_score


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
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1 Score (CV Average)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"optimal_k_{datetime.now().strftime('%m%d_%H%M%S')}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"The optimal number of neighbors is: {best_k} with F1 score: {cv_scores[best_k]:.4f}")
    return best_k, cv_scores


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = data_loader(which=3)
    train_data = train_data.T
    test_data = test_data.T

    # Fit PCA on training data (samples as rows, features as columns)
    # pca = PCA(n_components=9)
    # pca.fit(train_data)  # train_data shape: (n_samples, n_features)

    # reducer = umap.UMAP(n_components=12, random_state=0)
    # train_data_pca = reducer.fit_transform(train_data)
    # test_data_pca = reducer.transform(test_data)

    # Transform both train and test data using the same PCA object
    # train_data_pca = pca.transform(train_data)  # shape: (n_samples, 9)
    # test_data_pca = pca.transform(test_data)  # shape: (n_samples, 9)

    # find_optimal_k(train_data, train_label, k_range=range(1, 101, 2))

    knn = KNeighborsClassifier(n_neighbors=95)

    # Step 5: 訓練模型
    knn.fit(train_data, train_label)

    # Step 6: 預測
    y_pred = knn.predict(test_data)

    # Step 7: 評估模型表現
    precision, recall, f1_score = calculate_evaluation_metrics(test_label, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
