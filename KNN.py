from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
# import umap
# from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
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

    knn = KNeighborsClassifier(n_neighbors=85)

    # Step 5: 訓練模型
    knn.fit(train_data, train_label)

    # Step 6: 預測
    y_pred = knn.predict(test_data)

    # Step 7: 評估模型表現
    precision, recall, f1_score = calculate_evaluation_metrics(test_label, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    # # Step 8: 可視化結果（準確率 vs 鄰居數）
    # for dim in range(2, 21, 2):

    #     reducer = umap.UMAP(n_components=dim)
    #     train_data_pca = reducer.fit_transform(train_data)
    #     test_data_pca = reducer.fit_transform(test_data)

    #     accuracy_list = []
    #     k_values = range(1, 31, 2)  # 測試從 1 到 20 個鄰居數

    #     for k in k_values:
    #         knn = KNeighborsClassifier(n_neighbors=k)
    #         knn.fit(train_data_pca, train_label)
    #         y_pred = knn.predict(test_data_pca)
    #         accuracy_list.append(accuracy_score(test_label, y_pred))

    #     plt.figure(figsize=(10, 6))
    #     plt.plot(k_values, accuracy_list, marker="o")
    #     plt.title(f"K Values vs Model Accuracy with reduced dimension {dim}")
    #     plt.xlabel("K Values")
    #     plt.ylabel("Accuracy")
    #     plt.xticks(k_values)
    #     plt.grid()
    #     plt.savefig(f"fig/KvsModelAcc_{datetime.now().strftime('%m%d_%H%M%S')}.png", dpi=300, bbox_inches="tight")
    #     plt.show()
