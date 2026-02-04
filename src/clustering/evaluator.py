import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.clust_utils import batch_ari_computation


def parallel_ari(labels, predictions):
    # Parallelize the clustering process for each dataset
    color = '\033[92m'
    reset_color = '\033[0m'

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(batch_ari_computation, labels, predictions),
                            total=len(predictions),
                            bar_format=f"{color}{{l_bar}}{{bar}}{{r_bar}}{reset_color}"))
    return results


def process_files(dir_group: tuple) -> None:

    predictions_path, labels_path = dir_group
    predictions = np.load(predictions_path)
    labels = np.load(labels_path)

    # Parallelize the clustering process for each dataset
    print(f"Validation {os.path.basename(predictions_path)}")
    results = parallel_ari(labels, predictions)

    # Store the results in the predictions_iris array
    aris = np.asarray(results)

    # Get the parent directory
    parent_dir = os.path.dirname(os.path.dirname(predictions_path))

    # Create the predictions_iris folder if it doesn't exist
    ari_folder = os.path.join(parent_dir, 'ari')
    os.makedirs(ari_folder, exist_ok=True)

    # Generate the output file name
    filename = os.path.basename(predictions_path)
    filename_without_ext = os.path.splitext(filename)[0]
    output_filename = os.path.join(ari_folder, f'{filename_without_ext}_ari.npy')

    # Save the aris as a .npy file
    np.save(output_filename, aris)


def process_and_validate(dir_predictions: str, dir_labels: str):
    """
        Groups files with similar prefixes from the three directories provided. Assumes files end with
        '_predictions.npy', '_normalized.npy', and '_labels.npy'.

        Returns:
            A list of tuples, each containing the prefix and paths to the files in the form:
            [(prefix, normalized_file, prediction_file, label_file), ...]
        """

    parent_dir = os.path.dirname(dir_predictions)

    # List all .npy files in each directory
    prediction_files = [f for f in os.listdir(dir_predictions) if f.endswith('_predictions.npy')]
    label_files = [f for f in os.listdir(dir_labels) if f.endswith('_labels.npy')]

    # Extract the prefixes
    prediction_prefixes = {f.replace('_predictions.npy', ''):
                           os.path.join(directory_predictions, f) for f in prediction_files}
    label_prefixes = {f.replace('_labels.npy', ''):
                      os.path.join(directory_labels, f) for f in label_files}

    # Find common prefixes and group files
    common_prefixes = set(prediction_prefixes) & set(label_prefixes)

    for prefix in common_prefixes:
        filename = os.path.basename(prefix)
        filename_without_ext = os.path.splitext(filename)[0]
        output_filename = os.path.join(parent_dir, 'ari', f'{filename_without_ext}_predictions_ari.npy')
        if os.path.exists(output_filename):
            print(f"Skipping file {prefix} as ari_file already exist.")
            continue
        process_files((prediction_prefixes[prefix], label_prefixes[prefix]))


if __name__ == '__main__':
    # Example usage
    directory_predictions = 'data\\predictions'
    directory_labels = 'data\\labels'
    process_and_validate(directory_predictions, directory_labels)
