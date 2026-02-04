import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def normalize_file(file_path: str):
    """Load, normalize, and save the normalized .npy file."""
    # Load the .npy file
    data: np.ndarray = np.load(file_path)

    # Initialize an array to store the normalized data
    normalized_data: np.ndarray = np.zeros_like(data)
    color: str = '\033[92m'
    reset_color: str = '\033[0m'

    # Process each dataset with a progress bar
    with ProcessPoolExecutor() as executor:
        # Create a tqdm progress bar for the datasets within the file
        for i, result in enumerate(tqdm(executor.map(StandardScaler().fit_transform, data),
                                        total=len(data),
                                        desc=f"Normalizing {os.path.basename(file_path)}",
                                        bar_format=f"{color}{{l_bar}}{{bar}}{{r_bar}}{reset_color}")):
            normalized_data[i] = result

    # Get the parent directory
    parent_dir: str = os.path.dirname(os.path.dirname(file_path))

    # Create the normalized_data folder if it doesn't exist
    normalized_data_folder: str = os.path.join(parent_dir, 'normalized_data')
    os.makedirs(normalized_data_folder, exist_ok=True)

    # Generate the output file name
    filename: str = os.path.basename(file_path)
    filename_without_ext: str = os.path.splitext(filename)[0]
    output_filename: str = os.path.join(normalized_data_folder, f'{filename_without_ext}_normalized.npy')

    # Save the normalized data as a .npy file
    np.save(output_filename, normalized_data)


def normalize_directory_npy_data(directory: str):
    """Normalize all .npy files in the given directory that end with 'datasets.npy'."""
    # Normalize the directory path
    directory = os.path.normpath(directory)
    parent_dir: str = os.path.dirname(directory)

    # List all .npy files that end with 'datasets.npy' in the directory
    npy_files: list[str] = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

    for file_path in npy_files:
        filename: str = os.path.basename(file_path)
        filename_without_ext: str = os.path.splitext(filename)[0]
        output_filename: str = os.path.join(parent_dir, 'normalized_data', f'{filename_without_ext}_normalized.npy')
        if os.path.exists(output_filename):
            print(f"Skipping file {file_path} as normalized file already exist.")
            continue
        normalize_file(file_path)


if __name__ == '__main__':
    # directory of data
    path: str = '../datasets/raw_data'
    normalize_directory_npy_data(path)
