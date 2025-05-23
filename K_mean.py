import matplotlib.pyplot as plt
import numpy as np
from cuml import UMAP
from cuml.cluster import KMeans
from sklearn.metrics import silhouette_score

from main import data_loader

DIM = 2  # Set the dimension for UMAP
CLUSTER_NUM = 5  # Set the number of clusters for KMeans


def find_optimal_dimension_and_clusters(data, max_dim=20, max_clusters=20):
    dim_range = range(2, max_dim + 1)
    cluster_range = range(2, max_clusters + 1)
    scores = np.zeros((len(dim_range), len(cluster_range)))

    best_score = -1
    best_dim = None
    best_cluster = None

    for i, dim in enumerate(dim_range):
        umap_model = UMAP(n_components=dim, random_state=0)
        data = umap_model.fit_transform(train_data)
        for j, n_clusters in enumerate(cluster_range):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(data)
            try:
                score = silhouette_score(data, labels)
            except Exception:
                score = -1
            scores[i, j] = score
            if score > best_score:
                best_score = score
                best_dim = dim
                best_cluster = n_clusters
            print(f"UMAP dim={dim}, KMeans clusters={n_clusters}, Silhouette Score={score:.4f}")

    print(
        f"\nBest UMAP dim: {best_dim}, Best KMeans clusters: {best_cluster}, Best Silhouette Score: {best_score:.4f}"
    )
    # 畫熱力圖
    plt.figure(figsize=(8, 6))
    plt.imshow(
        scores,
        aspect="auto",
        origin="lower",
        extent=[min(cluster_range), max(cluster_range), min(dim_range), max(dim_range)],
        cmap="viridis",
    )
    plt.colorbar(label="Silhouette Score")
    plt.xlabel("KMeans n_clusters")
    plt.ylabel("UMAP n_components")
    plt.title("Silhouette Score Heatmap")
    plt.scatter([best_cluster], [best_dim], color="red", label="Best", marker="x")
    plt.legend()
    plt.show()


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
        plt.show()
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
        plt.show()
    else:
        print("Data is not 2D or 3D, skipping plot.")


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = data_loader(which=1)
    train_data = train_data.T
    find_optimal_dimension_and_clusters(train_data)
    # test_data = test_data.T
    # # Apply UMAP for dimensionality reduction
    # print("==========process UMAP==========")
    # umap_model = UMAP(n_components=DIM, random_state=0)
    # train_data_umap = umap_model.fit_transform(train_data)
    # test_data_umap = umap_model.transform(test_data)

    # # # Using the elbow method to find the optimal n_clusters
    # # find_optimal_clusters(train_data_umap, max_k=10)

    # print("==========process KMeans==========")
    # kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=0)
    # kmeans.fit(train_data_umap)

    # print("Cluster centers:\n", kmeans.cluster_centers_)

    # Plot the clusters
    # plot_clusters(train_data_umap, kmeans.labels_, kmeans.cluster_centers_)
