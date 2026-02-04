import numpy as np
from sklearn.metrics import adjusted_rand_score
from typing import Callable

def encoding_indices(index_values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(index_values > threshold, 1, 0)

def batch_ari_computation(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.asarray([adjusted_rand_score(labels, prediction) for prediction in predictions])

def batch_index_computation(labels: np.ndarray,
                            predictions: np.ndarray,
                            index: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    results = []
    for prediction in predictions:
        if len(np.unique(prediction)) <= 1:
            results.append(0)
        else:
            results.append(index(labels, prediction))
    return np.asarray(results)


def calculate_multiple_hot_ari(ari_matrix: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
        Generate a multi-hot encoded binary matrix by marking all ARI scores above the given threshold.

        Args:
            ari_matrix (np.ndarray): 2D array of shape (n_samples, n_algorithms) with ARI scores.
            threshold (float): Threshold above which scores are marked as 1.

        Returns:
            np.ndarray: Multi-hot encoded binary matrix of the same shape as ari_matrix.
        """
    return (ari_matrix > threshold).astype(int)
