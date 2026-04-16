import numpy as np
from typing import Tuple, Optional

from .model import model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(model):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight=None, **kwargs):


        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.pipeline.predict_proba(X)[:, 1].reshape(-1, 1)
        preds = (probs >= 0.5).astype(int).flatten()
        return preds, probs