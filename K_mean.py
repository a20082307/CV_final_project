import matplotlib.pyplot as plt
import numpy as np
from cuml import UMAP, KMeans
from cuml.metrics import confusion_matrix
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score

from main import data_loader

CLUSTER_NUM = 5  # Set the number of clusters for KMeans


def find_optimal_dimension_and_clusters(data, max_dim=100, max_clusters=100):
    dim_range = range(2, max_dim + 1)
    cluster_range = range(2, max_clusters + 1)
    scores = np.zeros((len(dim_range), len(cluster_range)))

    best_score = -1
    best_dim = None
    best_cluster = None

    for i, dim in enumerate(dim_range):
        print(f"==========process UMAP dim={dim}==========")
        umap_model = UMAP(n_components=dim)
        data = umap_model.fit_transform(train_data)
        for j, n_clusters in enumerate(cluster_range):
            print(f"==========process KMeans clusters={n_clusters}==========")
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(data)
            try:
                score = cython_silhouette_score(data, labels)
            except Exception:
                score = -1
            scores[i, j] = score
            if score > best_score:
                best_score = score
                best_dim = dim
                best_cluster = n_clusters
            print(f"UMAP dim={dim}, KMeans clusters={n_clusters}, Score={score:.4f}")

    print(f"\nBest UMAP dim: {best_dim}, Best KMeans clusters: {best_cluster}, Best Score: {best_score:.4f}")
    # 畫熱力圖
    plt.figure(figsize=(8, 6))
    plt.imshow(
        scores,
        aspect="auto",
        origin="lower",
        extent=[min(cluster_range), max(cluster_range), min(dim_range), max(dim_range)],
        cmap="viridis",
    )
    plt.colorbar(label="Score")
    plt.xlabel("KMeans n_clusters")
    plt.ylabel("UMAP n_components")
    plt.title("Score Heatmap")
    plt.scatter([best_cluster], [best_dim], color="red", label="Best", marker="x")
    plt.legend()
    plt.savefig("heatmap.png")


def silhouette_score_for_clusters(data, max_clusters=350):
    """
    Find the optimal number of clusters using silhouette score
    """
    scores = []
    cluster_range = range(2, max_clusters + 1)
    best_score = -1
    best_n_clusters = 2

    plt.figure(figsize=(10, 6))
    for n in cluster_range:
        print(f"Calculating silhouette score for n_clusters={n}")
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(data)
        try:
            score = cython_silhouette_score(data, labels)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_n_clusters = n
            print(f"n_clusters={n}, silhouette score={score:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette score for n_clusters={n}: {e}")
            scores.append(-1)

    plt.plot(cluster_range, scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Analysis (Best n_clusters: {best_n_clusters})")
    plt.axvline(x=best_n_clusters, color="r", linestyle="--", label=f"Best n_clusters: {best_n_clusters}")
    plt.legend()
    plt.savefig("silhouette_analysis.png")
    print(f"Best number of clusters: {best_n_clusters} with score: {best_score:.4f}")
    return best_n_clusters


def elbow_method_for_clusters(data):
    """
    Using the elbow method to find the optimal number of clusters
    """
    inertias = []
    cluster_range = range(1, 20)
    for n in cluster_range:
        print(f"process KMeans with n_clusters={n}")
        kmeans = KMeans(n_clusters=n, random_state=0)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(cluster_range, inertias, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow Method For Optimal k")
    plt.savefig("elbow.png")


def plot_clusters(data, labels, centers):
    """
    Plot the clusters
    """
    print("==========process plot==========")
    if data.shape[1] == 2:
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", alpha=0.5, label="Samples")
        plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centers")
        plt.title("KMeans Clustering Result (2D)")
        plt.legend()
        plt.savefig("clusters.png")

    elif data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", alpha=0.5)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="red",
            marker="X",
            s=200,
        )
        ax.set_title("KMeans Clustering Result (3D)")
        plt.savefig("clusters.png")

    else:
        print("Data is not 2D or 3D, skipping plot.")


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: precision, recall, and F1-score
    """
    # Get all true positives, false positives, and false negatives from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred).ravel().tolist()
    # Calculate global metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
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
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")

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


def predict():
    print("==========process KMeans==========")
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0)
    kmeans.fit(train_data)
    cluster_labels = kmeans.labels_

    # Predict clusters for test data
    print("==========process Prediction==========")
    pred_labels = kmeans.predict(test_data)

    # Calculate accuracy if we have ground truth labels
    if train_labels is not None and test_labels is not None:
        # Create a mapping from clusters to majority class
        cluster_to_label = {}
        for i in range(CLUSTER_NUM):
            mask = cluster_labels == i
            if np.any(mask):
                cluster_to_label[i] = np.bincount(train_labels[mask]).argmax()

        # Map predicted clusters to labels
        predicted_labels = np.array([cluster_to_label.get(label, -1) for label in pred_labels])

    return predicted_labels


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = data_loader(which=3)
    train_data = train_data.T
    test_data = test_data.T

    pred_labels = predict()
    accuracy, precision, recall, f1_score = calculate_evaluation_metrics(test_labels, pred_labels)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    ### Test Code ###
    # find_optimal_dimension_and_clusters(train_data)
    ### Using the elbow method to find the optimal n_clusters
    # elbow_method_for_clusters(train_data)
    # silhouette_score_for_clusters(train_data, max_clusters=350)
