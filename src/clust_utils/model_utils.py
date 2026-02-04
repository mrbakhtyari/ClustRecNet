from typing import List
import torch
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from pathlib import Path
import numpy as np
from src.training.models import (
    CNNResNetAttention,
    CNNResNet_NoAttention,
    ResNetAttention_NoCNN,
    CNNAttention_NoResidual,
    CNN_Only,
)
import time


def predict_algorithms(
    model: torch.nn.Module, x_padded: List[np.ndarray]
) -> np.ndarray:
    print("\n🔮 Predicting best algorithms using trained model...")
    x_tensor = torch.tensor(np.vstack(x_padded)).float()
    loader = DataLoader(
        TensorDataset(x_tensor), batch_size=len(x_tensor), shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch).cpu().numpy()
    end_time = time.time()
    print(f"⏱️ Inference time: {end_time - start_time:.4f} seconds")
    print("✅ Algorithm prediction complete.")
    return predictions


def build_model(model_name: str, path: Path) -> torch.nn.Module:
    print(f"\n📥 Loading model from: {path}")

    if model_name == "baseline_cnn":
        model = CNN_Only()
    elif model_name == "resnet":
        model = CNNResNetAttention()
    elif model_name == "no_att":
        model = CNNResNet_NoAttention()
    elif model_name == "no_res":
        model = CNNAttention_NoResidual()
    elif model_name == "no_cnn":
        model = ResNetAttention_NoCNN()
    else:
        raise ValueError(f"Unknown model type: {model_name}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded and set to eval mode.")
    return model


def load_models(model: torch.nn.Module, model_dir: str) -> List[torch.nn.Module]:
    """Load all models from a directory onto the specified device."""
    model_paths = sorted(glob(f"{model_dir}/*.pth"))
    print(f"🔍 Found {len(model_paths)} models in '{model_dir}'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in model_paths:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    return models


def predict_with_models(
    models: List[torch.nn.Module], data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Run ensemble prediction with all models and return prediction array (num_models, N, num_algorithms)."""
    all_predictions = []

    for model in models:
        with torch.no_grad():
            for (x_batch,) in data_loader:
                x_batch = x_batch.to(device)
                preds = model(x_batch).cpu().numpy()  # shape: (N, num_algorithms)
                all_predictions.append(preds)

    return np.array(all_predictions)  # shape: (num_models, N, num_algorithms)
