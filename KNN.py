from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from main import data_loader

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = data_loader()
    train_data = train_data.T
    test_data = test_data.T

    # Fit PCA on training data (samples as rows, features as columns)
    pca = PCA(n_components=9)
    pca.fit(train_data)  # train_data shape: (n_samples, n_features)

    # Transform both train and test data using the same PCA object
    train_data_pca = pca.transform(train_data)  # shape: (n_samples, 9)
    test_data_pca = pca.transform(test_data)  # shape: (n_samples, 9)

    knn = KNeighborsClassifier(n_neighbors=5)

    # Step 5: 訓練模型
    knn.fit(train_data_pca, train_label)

    # Step 6: 預測
    y_pred = knn.predict(test_data_pca)

    # Step 7: 評估模型表現
    print("準確率 (Accuracy):", accuracy_score(test_label, y_pred))
    print("\n分類報告 (Classification Report):\n", classification_report(test_label, y_pred))
    print("\n混淆矩陣 (Confusion Matrix):\n", confusion_matrix(test_label, y_pred))

    # Step 8: 可視化結果（準確率 vs 鄰居數）
    accuracy_list = []
    k_values = range(1, 21, 2)  # 測試從 1 到 20 個鄰居數

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data_pca, train_label)
        y_pred = knn.predict(test_data_pca)
        accuracy_list.append(accuracy_score(test_label, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_list, marker="o")
    plt.title("K Values vs Model Accuracy")
    plt.xlabel("K Values")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
    plt.grid()
    plt.savefig(f"fig/KvsModelAcc_{datetime.now().strftime("%m%d_%H%M%S")}.png", dpi=300, bbox_inches="tight")
    plt.show()
