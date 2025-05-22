import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from umap import UMAP

from main import data_loader

DIM = 2  # Set the dimension for UMAP
CLUSTER_NUM = 5  # Set the number of clusters for KMeans


def find_optimal_clusters(data, max_k=10):
    """
    Using the elbow method to find the optimal number of clusters
    """
    inertias = []
    cluster_range = range(1, max_k + 1)
    for n in cluster_range:
        print(f"process KMeans with n_clusters={n}")
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(cluster_range, inertias, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow Method For Optimal k")
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
    test_data = test_data.T
    # Apply UMAP for dimensionality reduction
    print("==========process UMAP==========")
    umap_model = UMAP(n_components=DIM, random_state=42)
    train_data_umap = umap_model.fit_transform(train_data)
    test_data_umap = umap_model.transform(test_data)

    # # Using the elbow method to find the optimal n_clusters
    # find_optimal_clusters(train_data_umap, max_k=10)

    print("==========process KMeans==========")
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=42)
    kmeans.fit(train_data_umap)

    print("Cluster centers:\n", kmeans.cluster_centers_)

    # Plot the clusters
    plot_clusters(train_data_umap, kmeans.labels_, kmeans.cluster_centers_)
