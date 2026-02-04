import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fixed input shape the recommendation model was trained on.
MAX_SAMPLES = 2000
MAX_FEATURES = 50


def fit_to_model_input(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shrink a dataset to <= (MAX_SAMPLES, MAX_FEATURES) for the recommender."""
    # Subsample rows.
    if X.shape[0] > MAX_SAMPLES:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=MAX_SAMPLES, replace=False)
        X = X[np.sort(idx)]

    # Reduce features via PCA.
    if X.shape[1] > MAX_FEATURES:
        X = PCA(n_components=MAX_FEATURES, random_state=seed).fit_transform(X)

    return X


def prepare_tensor(data: dict, names: list[str], seed: int = 0) -> list[np.ndarray]:
    padded = []
    for name in names:
        X = data[name]["X_standard"]
        X = fit_to_model_input(X, seed=seed)
        padded_X = pad_symmetric(X)
        padded.append(padded_X[np.newaxis, np.newaxis])
    return padded


def load_real_datasets(data_dir: Path) -> tuple[dict, list[str]]:
    """
    Load datasets from the specified directory.

    Args:
        data_dir (Path): Path to the directory containing dataset files.

    Returns:
        tuple: A dictionary of datasets and a list of dataset names.
    """
    dataset_files = sorted(data_dir.glob("*_dataset.npy"))
    dataset_names = [f.name.replace("_dataset.npy", "") for f in dataset_files]

    data = {}
    for name in dataset_names:
        X = np.load(data_dir / f"{name}_dataset.npy", allow_pickle=True)
        labels = np.load(data_dir / f"{name}_labels.npy", allow_pickle=True)

        X_std = StandardScaler().fit_transform(X)
        data[name] = {
            "X": X,
            "X_standard": X_std,
            "labels": labels,
        }
    return data, dataset_names


def extract_original_data(padded_data: np.ndarray) -> List[np.ndarray]:
    original_data_list = []

    for i in range(padded_data.shape[0]):
        slice_ = padded_data[i]

        # Determine the original shape by finding non-zero boundaries
        row_mask = np.any(slice_ != 0, axis=1)
        col_mask = np.any(slice_ != 0, axis=0)

        row_indices = np.where(row_mask)[0]
        col_indices = np.where(col_mask)[0]

        if row_indices.size > 0 and col_indices.size > 0:
            row_start, row_end = row_indices[0], row_indices[-1] + 1
            col_start, col_end = col_indices[0], col_indices[-1] + 1

            original_data = slice_[row_start:row_end, col_start:col_end]
        else:
            original_data = np.array([[]])  # Empty array if no original data is found

        original_data_list.append(original_data)

    return original_data_list


def scale_datasets(datasets: List[np.ndarray]) -> np.ndarray:
    return np.asarray([StandardScaler().fit_transform(dataset) for dataset in datasets])


def load_npy_files(folder: str) -> Tuple[Dict[str, np.ndarray]]:
    """
    Load multiple .npy files from a directory with specific filenames.

    Args:
        folder (str): Directory containing the .npy files.

    Returns:
        Tuple containing:
            - Dict[str, np.ndarray]: Loaded data arrays.
            - Dict[str, str]: Mapping of logical names to filenames.
    """
    files = {
        "features": "features.npy",
        "labels": "ground_truth_labels.npy",
        "predictions": "algorithm_predictions.npy",
        "ari": "ari_scores.npy",
    }
    data = {}
    for key, filename in files.items():
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        data[key] = np.load(path, allow_pickle=True)
        print(f"🔹 Loaded {filename}: {len(data[key])} entries")
    return data


def save_object_array(array_slice, path):
    result = np.empty(len(array_slice), dtype=object)
    for i, item in enumerate(array_slice):
        result[i] = item
    np.save(path, result)


def shuffle_and_split(
    data_dict: Dict[str, np.ndarray],
    out_dir: str,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> None:
    """
    Shuffle and split the real-world clustering data into train and test folders.

    The four expected keys in data_dict are:
        - 'features'
        - 'labels'
        - 'predictions'
        - 'ari'

    Files are saved under `train/` and `test/` subfolders with original filenames:
        - features.npy
        - ground_truth_labels.npy
        - algorithm_predictions.npy
        - ari_scores.npy

    Args:
        data_dict (Dict[str, np.ndarray]): Dictionary of real-world clustering arrays.
        out_dir (str): Root output folder to contain train/ and test/ folders.
    """
    os.makedirs(out_dir, exist_ok=True)

    combined = list(
        zip(
            data_dict["features"],
            data_dict["labels"],
            data_dict["predictions"],
            data_dict["ari"],
        )
    )

    np.random.seed(seed)
    np.random.shuffle(combined)

    features, labels, predictions, ari = zip(*combined)

    total = len(features)
    split_index = int(total * test_ratio)

    test_slice = slice(0, split_index)
    train_slice = slice(split_index, total)

    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "test"), exist_ok=True)

    # Save test set
    save_object_array(
        features[test_slice], os.path.join(out_dir, "test", "features.npy")
    )
    save_object_array(
        labels[test_slice], os.path.join(out_dir, "test", "ground_truth_labels.npy")
    )
    save_object_array(
        predictions[test_slice],
        os.path.join(out_dir, "test", "algorithm_predictions.npy"),
    )
    save_object_array(ari[test_slice], os.path.join(out_dir, "test", "ari_scores.npy"))

    # Save train set
    save_object_array(
        features[train_slice], os.path.join(out_dir, "train", "features.npy")
    )
    save_object_array(
        labels[train_slice], os.path.join(out_dir, "train", "ground_truth_labels.npy")
    )
    save_object_array(
        predictions[train_slice],
        os.path.join(out_dir, "train", "algorithm_predictions.npy"),
    )
    save_object_array(
        ari[train_slice], os.path.join(out_dir, "train", "ari_scores.npy")
    )

    print(f"✅ Split completed → train: {total - split_index}, test: {split_index}")


def load_and_preprocess_data(
    data_path: str,
    ari_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess raw clustering data and ari of all 10 clustering data.

    Returns:
        - dataset: np.ndarray of shape (N, 1, 2000, 50)
        - labels: np.ndarray of shape (N, 10)
    """
    raw_data = np.load(data_path, allow_pickle=True)
    ari = np.load(ari_path, allow_pickle=True)
    ari_array = np.stack(ari, axis=0)
    labels = (ari_array > 0.8).astype(int)
    padded_data = [pad_symmetric(sample).reshape(1, 2000, 50) for sample in raw_data]

    return padded_data, labels


def pad_symmetric(data):
    max_height = MAX_SAMPLES
    max_width = MAX_FEATURES
    padded = np.zeros((max_height, max_width), dtype=np.float64)

    img_height = data.shape[0]
    img_width = data.shape[1]
    height_diff = max_height - img_height
    width_diff = max_width - img_width

    height_pad_top = height_diff // 2
    width_pad_left = width_diff // 2

    # Apply padding directly to the pre allocated array
    padded[
        height_pad_top : height_pad_top + img_height,
        width_pad_left : width_pad_left + img_width,
    ] = data.astype(np.float64)

    return padded
