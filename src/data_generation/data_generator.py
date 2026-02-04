import numpy as np
import repliclust
from typing import Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


# Refer to Section 2.1 "Data Generation" in the paper for details on the configuration and 
# implementation of synthetic data generation strategies.


@dataclass
class DataConfig:
    num_clusters: int
    num_samples: int
    num_dimensions: int


class DataGenerationStrategy(ABC):
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data using the strategy."""
        pass


class DataGenerator:
    """Data generator class."""

    def __init__(self, strategy: DataGenerationStrategy) -> None:
        """Initialize the data generator with a strategy."""
        self.strategy = strategy

    def generate_dataset(self, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get a dataset using the strategy and shuffle."""
        data, labels = self.strategy.generate(config)
        indices = np.random.permutation(data.shape[0])
        return data[indices], labels[indices]

    def generate_datasets(self, config: dict, num_datasets: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a predefined number of realizations using the strategy."""
        datasets_labels = [self.generate_dataset(config) for _ in range(num_datasets)]
        datasets, labels = zip(*datasets_labels)
        return np.asarray(datasets), np.asarray(labels)


class CesarCominStrategy(DataGenerationStrategy):
    """Cesar Comin data generation strategy from https://doi.org/10.1371%2Fjournal.pone.0210236."""

    def generate(self, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset using the Cesar Comin method.
        input:
            - alpha: parameter to control the dispersion of the instances (the distance between the classes)
        """
        alpha = config['alpha']
        e, v, e_bar = 0., 0.25, 1.  # parameters to control the data generation process. No need to change them

        m = int(np.round((e_bar ** 2. - e ** 2.) / v))
        e_chap = np.sqrt(e / m)
        v_chap = -e_chap ** 2. + np.sqrt(e_chap ** 4. + v / m)

        dataset = []
        labels = []
        n_i = self.config.num_samples // self.config.num_clusters
        remainder = self.config.num_samples % self.config.num_clusters

        for i in range(self.config.num_clusters):
            q = np.random.normal(size=[self.config.num_dimensions, m])
            f = (e_chap + np.sqrt(v_chap) * q) / alpha
            u = 2. * (np.random.rand(self.config.num_dimensions) - 0.5)

            size = n_i + 1 if i < remainder else n_i
            z = np.random.normal(size=[m, size])
            y = np.repeat(i, size)
            x = np.dot(f, z).T + u

            dataset.append(x)
            labels.append(y)

        return np.concatenate(dataset), np.concatenate(labels)


class RepliclustStrategy(DataGenerationStrategy):
    """Repliclust data generation strategy from https://arxiv.org/abs/2303.14301v1"""

    def _create_data_generator(self, config: dict) -> repliclust.DataGenerator:
        """
        config (dict): Configuration for the Repliclust data generator.
        radius_maxmin: float, >=1
            Ratio between the maximum and minimum radii among all clusters in a mixture model.
        aspect_maxmin: float, >=1
            Ratio between the maximum and minimum aspect ratios among all clusters in a mixture model.
        aspect_ref: float, >=1
            Typical aspect ratio for the clusters in a mixture model. For example, if aspect_ref = 10,
             we expect that all clusters in the mixture model are strongly elongated.
        imbalance_maxmin: float, >=1
            Ratio between the greatest and smallest group sizes among all clusters in the mixture model.
        min_overlap: float in (0,1)
            The minimum required overlap between a cluster and *some* other cluster.
            This minimum overlap allows you to guarantee that no cluster will be isolated from all other clusters.
        max_overlap: float in (0,1)
            The maximum allowed level of overlap between any two clusters.
            Measured as the fraction of cluster volume that overlaps.
        scale: float
            Reference length scale for generated data
        distributions: list of [str | tuple[str, dict]]
            Selection of probability distributions that should appear in  each mixture model.
            Format is a list in which each element is either the name of the probability distribution OR a tuple whose
             first entry is the name and the second entry is a dictionary of distributional parameters.
            To print the names of all supported distributions and their parameters (along with default values),
             print the output of repliclust.get_supported_distributions().
        distributions_proportions:
            The proportions of clusters that have each distribution listed in `distributions`.
        mode : {"auto", "lda", "c2c"}
            Select the degree of precision when computing cluster overlaps.
        """
        """Configure an archetype for the Repliclust data generator."""
        archetype = repliclust.Archetype(
            n_clusters=self.config.num_clusters,
            dim=self.config.num_dimensions,
            n_samples=self.config.num_samples,
            **config
        )
        repliclust.set_seed(config.get('r_seed', 0))
        return repliclust.DataGenerator(archetype=archetype)

    def generate(self, config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a dataset using repliclust."""
        data_generator = self._create_data_generator(config)
        dataset, labels, _ = data_generator.synthesize(quiet=True)
        return dataset, labels


if __name__ == '__main__':

    # Configuration for data generation
    config = DataConfig(
        num_clusters=2,
        num_samples=500,
        num_dimensions=2
    )

    # Example of generating datasets with repliclust data generation method
    repliclust_config = {
        'min_overlap': 0.001,
        'max_overlap': 0.002,
        'aspect_ref': 1,
        'aspect_maxmin': 1,
        'radius_maxmin': 1,
        'distributions': ['normal'],
        'imbalance_ratio': 1
    }
    repliclust_strategy = RepliclustStrategy(config)
    repliclust_generator = DataGenerator(repliclust_strategy)
    repliclust_data, repliclust_labels = repliclust_generator.generate_dataset(repliclust_config)
    print(f"Repliclust data shape:{repliclust_data.shape}")
    print(f"Repliclust labels shape:{repliclust_labels.shape}")


    # Example of generating datasets with Cesar Comin data generation method
    cesar_comin_config = {
        'alpha': 10
    }
    cesar_comin_strategy = CesarCominStrategy(config)
    cesar_comin_generator = DataGenerator(cesar_comin_strategy)
    cesar_comin_data, cesar_comin_labels = cesar_comin_generator.generate_dataset(cesar_comin_config)
    print(f"CesarComin data shape:{cesar_comin_data.shape}")
    print(f"CesarComin labels shape:{cesar_comin_labels.shape}")
