from datetime import datetime

import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from main import data_loader


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: precision, recall, and F1-score
    """
    # Get all true positives, false positives, and false negatives from confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate global precision, recall, and F1-score
    # True positives are the sum of diagonal elements
    tp_sum = sum(cm[i, i] for i in range(len(cm)))
    # Actual positives (row sums)
    actual_pos = cm.sum(axis=1)
    # Predicted positives (column sums)
    pred_pos = cm.sum(axis=0)

    # Calculate global metrics
    precision = tp_sum / pred_pos.sum() if pred_pos.sum() > 0 else 0
    recall = tp_sum / actual_pos.sum() if actual_pos.sum() > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix:\n", cm)

    return precision, recall, f1_score


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = data_loader()
    train_data = train_data.T
    test_data = test_data.T

    # Fit PCA on training data (samples as rows, features as columns)
    pca = PCA(n_components=9)
    pca.fit(train_data)  # train_data shape: (n_samples, n_features)

    reducer = umap.UMAP(n_components=12)
    train_data_pca = reducer.fit_transform(train_data)
    test_data_pca = reducer.fit_transform(test_data)

    # Transform both train and test data using the same PCA object
    # train_data_pca = pca.transform(train_data)  # shape: (n_samples, 9)
    # test_data_pca = pca.transform(test_data)  # shape: (n_samples, 9)

    knn = KNeighborsClassifier(n_neighbors=29)

    # Step 5: 訓練模型
    knn.fit(train_data_pca, train_label)

    # Step 6: 預測
    y_pred = knn.predict(test_data_pca)

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
