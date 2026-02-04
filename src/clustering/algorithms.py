import numpy as np
from sklearn.neighbors import kneighbors_graph
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, HDBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
# from src.clust_utils import ignore_warnings
from typing import Optional


# This file contains clustering algorithms introduced in Section "2.2 Clustering Algorithms"
# Their configurations are saved in configs folder with yaml format.


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        pass


class KMedians(ClusteringAlgorithm):
    """
    K-Medians clustering algorithm with support for multiple initializations,
    empty cluster handling, verbosity, and limited iterations.

    Attributes:
        n_clusters (int): Number of clusters.
        n_init (int): Number of initializations to perform.
        max_iter (int): Maximum number of iterations per run.
        random_state (Optional[int]): Seed for reproducibility.
        verbose (bool): If True, print intermediate logs.
    """
    def __init__(self, n_clusters: int, n_init: int = 100, max_iter: int = 300, verbose: bool = False, random_state: int = 0):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.max_iter = max_iter
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the K-Medians model to the given data.

        Args:
            data (np.ndarray): Data array of shape (n_samples, n_features).
        """
        np.random.seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        best_median = None
        best_labels = None
        best_inertia = np.inf

        for init_num in range(self.n_init):
            # Randomly initialize medians
            medians = data[rng.choice(data.shape[0], self.n_clusters, replace=False)]
            labels = np.zeros(data.shape[0], dtype=int)

            for iteration in range(self.max_iter):
                ## Assign each point to the nearest median using L1 distance
                distances = np.abs(data[:, np.newaxis] - medians).sum(axis=2)
                new_labels = np.argmin(distances, axis=1)

                # Update medians
                new_medians = []
                for i in range(self.n_clusters):
                    if np.any(new_labels == i):
                        cluster_data = data[new_labels == i]
                        median = np.median(cluster_data, axis=0)
                    else:
                        # Handle empty cluster: reinitialize to random point
                        median = data[rng.integers(data.shape[0])]
                    new_medians.append(median)
                new_medians = np.array(new_medians)

                if np.all(new_labels == labels):
                    break

                medians = new_medians
                labels = new_labels

            # Calculate total L1-distance (inertia)
            inertia = np.sum([np.abs(data[labels == i] - medians[i]).sum() for i in range(self.n_clusters)])
            
            if self.verbose:
                print(f"[Init {init_num}] Inertia: {inertia:.4f}")

            if inertia < best_inertia:
                best_median = medians
                best_labels = labels
                best_inertia = inertia

        self.cluster_centers_ = best_median
        self.labels_ = best_labels

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            data (np.ndarray): Data to cluster.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self.cluster_centers_ is None:
            raise Exception("Model has not been fitted yet.")
        distances = np.abs(data[:, np.newaxis] - self.cluster_centers_).sum(axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the model to data and return the cluster assignments.

        Args:
            data (np.ndarray): Data to cluster.

        Returns:
            np.ndarray: Cluster labels.
        """
        self.fit(data)
        return self.labels_


class ClusteringAlgorithmFactory:

    def __init__(self):
        self.algorithm: dict[str, callable] = {
            "kmeans": KMeans,
            "kmedians": KMedians,
            "spectral_clustering": SpectralClustering,
            "ward": AgglomerativeClustering,
            "agglomerative": AgglomerativeClustering,
            "dbscan": DBSCAN,
            "hdbscan": HDBSCAN,
            "optics": OPTICS,
            "birch": Birch,
            "gaussian": GaussianMixture,
        }

    def create_algorithm(self, algorithm_name: str, config: dict) -> ClusteringAlgorithm:
        if algorithm_name not in self.algorithm:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return self.algorithm[algorithm_name](**config)


class ClusteringAlgorithms:
    def __init__(self, clustering_setup: dict[str, dict]):
        self.algorithm_factory = ClusteringAlgorithmFactory()
        self.clustering_setup = clustering_setup
        self.num_algorithms = len(clustering_setup)

    def cluster(self, data: np.ndarray) -> np.ndarray:
        results = []
        for algorithm_name, config in self.clustering_setup.items():
            algorithm = self.creator(data, algorithm_name, config)
            prediction = self.predictor(algorithm, data)
            results.append(prediction)
        return np.asarray(results)

    def creator(self, data, algorithm_name, config):
        if "n_neighbors" in config:
            config["connectivity"] = self.get_connectivity_graph(data, config["n_neighbors"])
            del config["n_neighbors"]
        algorithm = self.algorithm_factory.create_algorithm(algorithm_name, config)
        return algorithm

    # @ignore_warnings
    def predictor(self, algorithm, data):
        return algorithm.fit_predict(data)

    @staticmethod
    def get_connectivity_graph(data, n_neighbors):
        if n_neighbors is None:
            return None
        else:
            connectivity = kneighbors_graph(data, n_neighbors=n_neighbors, include_self=False)
            return 0.5 * (connectivity + connectivity.T)
