import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os
import torch


def save_data(dataset: np.ndarray, labels: np.ndarray, path: str, file_name: str) -> None:
    np.save(f"{path}{file_name}_datasets.npy", dataset)
    np.save(f"{path}{file_name}_labels.npy", labels)

def concatenate_and_save(files_path: str, output_file_name: str) -> None:
    files = [os.path.join(files_path, f) for f in os.listdir(files_path)]
    datasets = [np.load(file) for file in files]
    data = np.asarray(datasets).reshape(-1, datasets[0].shape[-1])
    sparse_matrix = csr_matrix(data)
    save_npz(output_file_name, sparse_matrix)


def save_model(model: object, path: str):
    model_core, _, _ = model
    torch.save(model_core.state_dict(), path)
