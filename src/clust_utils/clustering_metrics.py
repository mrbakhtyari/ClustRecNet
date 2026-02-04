import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from typing import Union, Callable


def dunn_score(data: np.ndarray,
               labels: np.ndarray,
               metric: Union[str, Callable] = 'euclidean') -> float:
    """
    Calculates the Dunn index for the given clustering.

    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        A list of n_features-dimensional data points. Each row corresponds to a single data point.

    labels : array-like of shape (n_samples,). Predicted labels for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by sklearn.metrics.pairwise.pairwise_distances.
        If metric is a callable function, it is called on each pair of instances (rows) and the result recorded.
         The callable should take two arrays as input and return one value indicating the distance between them.

    Returns:
    --------
    float: The Dunn index for the given clustering.
    """
    distances = pairwise_distances(data, metric=metric)
    return _calculate_dunn_from_distances(labels, distances)


def normalize_to_smallest_integers(labels):
    """Normalizes a list of integers so that each number is reduced to the minimum possible integer,
     maintaining the order of elements."""
    unique_labels = np.unique(labels)
    return np.searchsorted(unique_labels, labels)


def _calculate_dunn_from_distances(labels, distances):
    """Calculates the Dunn index given the cluster labels and pairwise distances."""
    labels = normalize_to_smallest_integers(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        raise ValueError("Dunn Index is undefined for fewer than 2 clusters.")

    if distances.shape[0] != len(labels):
        raise ValueError("Labels and distance matrix shape mismatch.")

    # Min inter-cluster distances
    min_intercluster = np.inf
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                dist = distances[i, j]
                if dist < min_intercluster:
                    min_intercluster = dist

    # Max intra-cluster distance (cluster diameters)
    max_diameter = 0.0
    for cluster_id in unique_labels:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < 2:
            continue
        intra_dists = distances[np.ix_(idx, idx)]
        local_diameter = np.max(intra_dists)
        if local_diameter > max_diameter:
            max_diameter = local_diameter

    if max_diameter == 0:
        raise ValueError("Max diameter is zero. Likely all clusters are singletons.")

    return min_intercluster / max_diameter


# Example usage:
if __name__ == '__main__':

    X = np.array([
        [0, 0], [0, 1], [1, 0],
        [10, 10], [10, 11], [11, 10]
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    score = dunn_score(X, labels)
    print(score)  # Should print a Dunn index value, 9.51
