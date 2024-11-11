import numpy as np
import pickle
import joblib
from pathlib import Path
import os


class Classifier:
    def __init__(self):
        """
        You can load the model as a member variable while instantiation the classifier
        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.model = pickle.load(open(Path(root_path, 'classification_model.pkl'), 'rb'))
        self.mean = pickle.load(open(Path(root_path, 'classification_mean.pkl'), 'rb'))
        self.std_dev = pickle.load(open(Path(root_path, 'classification_std.pkl'), 'rb'))


    def inference(self, X: np.array) -> int:
        """
        Inference a single data
        Args:
            X: The feature vector with dim=256 of the data which needs to be classified

        Returns:
            Classification result, the index of the category.
        """
        return self.model.predict((np.array([X]) - self.mean) / self.std_dev)[0]
