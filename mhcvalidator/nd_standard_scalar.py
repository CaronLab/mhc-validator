from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
import numpy as np


class NDStandardScaler(TransformerMixin):
    """
    Standard scale features in an array of arbitrary dimensions.
    https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
    """
    def __init__(self):
        self._scaler = MinMaxScaler(copy=True)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X