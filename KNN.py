import numpy as np


class KNearestNeighbors:
    """A simple implementation of the K-Nearest Neighbors (KNN) classifier.

    Attributes:
        k (int, default=5): Number of nearest neighbors to consider.
        X_train (array-like): The training data.
        y_train (array-like): The class labels for the training data.
    """

    def __init__(self, k=5):
        """Initialize the KNN classifier with the number of neighbors.

        Args:
            k (int, optional, default=5): Number of nearest neighbors to consider.
        """
        self.k = k

    def fit(self, X, y):
        """Fit the KNN classifier to the training data.

        Args:
            X (array-like): The training data.
            y (array-like): The class labels for the training data.
        """
        # lazy learning
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (array-like): The input data to predict.

        Returns:
            ndarray: Predicted class labels for each sample in X.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def score(self, X, y):
        """Compute the accuracy of the classifier on the provided data.

        Args:
            X (array-like): The input data to score.
            y (array-like): True class labels for the input data.

        Returns:
            float: The accuracy of the classifier on the provided data.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def _predict(self, x):
        """Predict the class label for a single sample.

        Args:
            x (array-like): The input data to predict.

        Returns:
            int: The predicted class label for the input data.
        """
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
