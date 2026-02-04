import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from algorithms import ClusteringAlgorithms


def parallel_clustering(data: np.ndarray, clustering_algo: ClusteringAlgorithms) -> list[np.ndarray]:

    color = '\033[92m'
    reset_color = '\033[0m'

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(clustering_algo.cluster, data),
                            total=len(data),
                            bar_format=f"{color}{{l_bar}}{{bar}}{{r_bar}}{reset_color}"))
    return results


def process_file(file_path: str, clustering_algo: ClusteringAlgorithms) -> np.ndarray:
    """Process a single .npy file and save the predictions_iris."""
    # Load the .npy file
    data = np.load(file_path)

    # Initialize an array to store the predictions_iris
    predictions = np.zeros((data.shape[0], clustering_algo.num_algorithms, data.shape[1]), dtype=np.int8)

    # Parallelize the clustering process for each dataset
    print(f"Clustering {os.path.basename(file_path)}")
    results = parallel_clustering(data, clustering_algo)

    # Store the results in the predictions_iris array
    for i, result in enumerate(results):
        predictions[i] = result

    save_predictions(predictions, file_path)

    return predictions


def save_predictions(predictions: np.ndarray, file_path: str) -> None:
    # Get the parent directory
    parent_dir = os.path.dirname(os.path.dirname(file_path))

    # Create the predictions_iris folder if it doesn't exist
    predictions_folder = os.path.join(parent_dir, 'predictions_iris')
    os.makedirs(predictions_folder, exist_ok=True)

    # Generate the output file name
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    output_filename = os.path.join(predictions_folder, f'{filename_without_ext}_predictions.npy')
    # Save the predictions_iris as a .npy file
    np.save(output_filename, predictions)


def process_and_cluster(directory: str, clustering_algo: ClusteringAlgorithms):
    """Process .npy files with 'datasets_normalized.npy' in their name and save clustering results."""
    # Normalize the directory path
    parent_dir = os.path.dirname(directory)
    directory = os.path.normpath(directory)

    # List all .npy files that end with 'datasets_normalized.npy' in the directory
    npy_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('normalized.npy')]

    # Process each file
    for file_path in npy_files:
        # Check if the output file already exists
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]
        output_filename = os.path.join(parent_dir, 'predictions', f'{filename_without_ext}_predictions.npy')
        if os.path.exists(output_filename):
            print(f"Skipping file {file_path} as predictions file already exist.")
            continue
        process_file(file_path, clustering_algo)
