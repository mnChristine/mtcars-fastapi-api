"""
app/model_utils.py
------------------
Lightweight linear regression class used when scikit-learn is not available.
The class is also required at load time if the pre-generated model.pkl
was saved without sklearn (e.g. the bundled artifact in this repo).

When the user runs scripts/train_model.py with sklearn installed, the saved
model will be a proper sklearn.linear_model.LinearRegression object instead.
"""

import numpy as np


class LinearRegressionModel:
    """
    Minimal sklearn-compatible linear regression implemented with NumPy.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_features,)
    intercept_ : float
    """

    def __init__(self, coef_: np.ndarray, intercept_: float):
        self.coef_ = np.asarray(coef_, dtype=float)
        self.intercept_ = float(intercept_)

    def predict(self, X) -> np.ndarray:
        """Return predicted values for input array X (shape n_samples × n_features)."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def __repr__(self) -> str:
        return (
            f"LinearRegressionModel("
            f"coef_={self.coef_}, intercept_={self.intercept_:.4f})"
        )
