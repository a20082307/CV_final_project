import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cuml.cluster import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from cuml.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from main import data_loader

# 假設你的數據
X, y, X_test, y_test = data_loader(which=3)
X = X.T
y = y.T


# 定義 KNN 和 KMeans 的驗證範圍
n_neighbors_range = range(3, 201, 2)  # 測試 1 到 50 個鄰居
results = []

for n_neighbors in n_neighbors_range:
    print(f"Processing n_neighbors={n_neighbors}...")
    # Step 1: 驗證 KNN
    print("Step 1: Validating KNN...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # 使用 GPU 加速的 KNN
    knn_cross_val_score = cross_val_score(knn, X.copy(), y.copy(), cv=5, scoring="accuracy").mean()
    print(f"KNN Cross Validation Score: {knn_cross_val_score:.4f}")

    # Step 2: 計算對應的 KMeans 群集數
    print("Step 2: Validating KMeans...")
    n_clusters = max(2, 35000 // n_neighbors)  # 確保群集數至少為 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    # Step 3: 驗證 KMeans
    # 計算 Silhouette Score（需要計算樣本到群集中心的平均距離）
    silhouette_avg = cython_silhouette_score(X, labels)  # 近似 Silhouette 分數（越高越好）
    print(f"KMeans Silhouette Score: {silhouette_avg:.4f}")

    # Step 4: 記錄結果
    results.append(
        {
            "n_neighbors": n_neighbors,
            "knn_cross_val_score": knn_cross_val_score,
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
        }
    )

# 將結果轉換為 DataFrame，方便查看
results_df = pd.DataFrame(results)

# 找到最佳組合
best_result = results_df.loc[results_df["silhouette_score"].idxmax()]  # 以 Silhouette Score 為主導
print("最佳結果：")
print(best_result)

# 畫圖：雙 Y 軸
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左側 Y 軸：KNN 的平均距離
ax1.set_xlabel("n_neighbors (KNN)")
ax1.set_ylabel("KNN Cross Validation Score (the higher the better)", color="blue")
ax1.plot(
    results_df["n_neighbors"], results_df["knn_cross_val_score"], label="KNN Cross Validation Score", color="blue"
)
ax1.tick_params(axis="y", labelcolor="blue")

# 右側 Y 軸：KMeans 的 Silhouette Score
ax2 = ax1.twinx()
ax2.set_ylabel("Silhouette Score (the higher the better)", color="green")
ax2.plot(results_df["n_neighbors"], results_df["silhouette_score"], label="Silhouette Score", color="green")
ax2.tick_params(axis="y", labelcolor="green")

# 添加圖例
fig.tight_layout()
plt.title("KNN Cross Validation Score and KMeans Silhouette Score vs n_neighbors")
plt.savefig("knn_kmeans_analysis.png")
