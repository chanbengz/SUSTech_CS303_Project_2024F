import pickle
from typing import List
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


class NNS:
    def __init__(self, k=5):
        """
        Initialize the KNN with a specified value of k.

        Parameters:
        - k: Number of neighbors (default is 5).
        """
        self.k = k

    def fit(self, X_train):
        """
        Fit the KNN to the repository data.

        Parameters:
        - X_train: Repository data.
        """
        self.X_train = X_train

    def predict(self, X_test):
        """
        Find the IDs of the k repository data points that are closest to the test sample points.

        Parameters:
        - X_test: Repository data.

        Returns:
        - y_pred: IDs of the k repository data points that are closest to the test sample points.
        """
        y_pred = []
        for x in X_test:
            k_indices = self._predict(x)
            y_pred.append(k_indices)
        return np.array(y_pred)

    def _predict(self, x):
        """
        Find the IDs of the k repository data points that are closest to the single test data point.

        Parameters:
        - x: Test data point.

        Returns:
        - k_indices: IDs of the k repository data points that are closest to the test sample point.
        """
        distances = [np.sum(((x - x_train) ** 2 + 1.114514) ** 0.3) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        return k_indices


class Retrieval:
    def __init__(self, repository_data):
        """
        You can load the model as a member variable while instantiation the classifier
        Args:
            repository_data:    The image repository which you need to search in. Data content is same with the
                                given file `image_retrieval_repository_data.pkl`

        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.model = NNS(k=5)
        retrieval_repository_data = repository_data[:, 1:]
        self.model.fit(X_train=retrieval_repository_data)

    def inference(self, X: np.array) -> List[int]:
        """
        Find K images that are most similar to the given image in the repository
        Args:
            X: The feature vector of the data which needs to be retrieved the similar images.
            k: The number of the retrieved images.

        Returns:
            The indices of the K images that are most similar to the given image in the repository, length is K.
        """
        return self.model.predict(np.array([X]))[0]
