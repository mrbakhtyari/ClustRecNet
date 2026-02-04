from enum import Enum


class Algorithm(Enum):
    KMEANS = ("KMeans", "k-means", "k_means")
    KMEDIANS = ("KMedians", "k-medians", "k_medians")
    SPECTRAL = ("SpectralClustering", "Spectral\nClustering", "spectral_clustering")
    WARD = ("Ward", "Ward", "ward")
    AGGLOMERATIVE = ("AgglomerativeClustering", "Agglomerative\nClustering", "agglomerative_clustering")
    DBSCAN = ("DBSCAN", "DBSCAN", "dbscan")
    HDBSCAN = ("HDBSCAN", "HDBSCAN", "hdbscan")
    OPTICS = ("OPTICS", "OPTICS", "optics")
    BIRCH = ("BIRCH", "BIRCH", "birch")
    GAUSSIAN_MIXTURE = ("GaussianMixture", "Gaussian Mixture", "gaussian_mixture")

    def __init__(self, sklearn_name: str, display_name: str, internal_name: str):
        self.sklearn_name = sklearn_name
        self.display_name = display_name
        self.internal_name = internal_name


ALGORITHMS = list(Algorithm)
ALGORITHMS_SKLEARN = [alg.sklearn_name for alg in ALGORITHMS]
ALGORITHMS_DISPLAY = [alg.display_name for alg in ALGORITHMS]
ALGORITHMS_INTERNAL = [alg.internal_name for alg in ALGORITHMS]