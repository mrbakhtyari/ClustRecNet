import os
import numpy as np
import torch
from src.clust_utils import plot_results, load_and_preprocess_data, save_model, set_seed, load_npy_files, shuffle_and_split
from src.training.kfold_runner import run_kfold_training
from torch.utils.data import TensorDataset
from datetime import datetime


def train_pipeline_main(
    data_path: str,
    seed: int = 0,
    model_name: str = "resnet",
    epochs: int = 30,
    run_name: str = ""
) -> None:
    """
    Main entry point for training pipeline.

    Args:
        data_path (str): Path to dataset.
        seed (int): Random seed for reproducibility.
        model_type (str): Model architecture to use ('baseline_cnn', 'resnet', etc.).
        epochs (int): Number of training epochs.
        run_name (Optional[str]): Optional name for logging or tracking the run.
    """
    # Setup device and paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(seed)
    print("🔧 Environment Info:")
    print(f"Python version: {os.sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA device: {device}")
    print(f"seed: {seed}")

    # 🔒 Ensuring full reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    
    # 🔒 Avoiding TF32 behavior
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    

    # Create directories for artifacts
    os.makedirs("artifacts/metrics", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"✅ Timestamp for this run: {timestamp}")

    # Define dataset name and file prefix
    file_prefix = f"{run_name}_seed{seed}_{timestamp}"

    # Step 1: Load and split test/train data
    data_dict = load_npy_files(data_path)
    shuffle_and_split(data_dict, out_dir=data_path, test_ratio=0.1, seed=seed)
    print("✅ Data loaded and split into train/test sets.")
    
    features_path = os.path.join(data_path, "train", "features.npy")
    ari_path = os.path.join(data_path, "train", "ari_scores.npy")
    print(f"✅ Loading data from: {features_path}")
    print(f"✅ Loading labels from: {ari_path}")
    datasets, labels = load_and_preprocess_data(features_path, ari_path)

    # Efficient conversion
    datasets_np = np.array(datasets)
    labels_np = np.array(labels)

    # Step 3: Create PyTorch datasets
    train_tensor = TensorDataset(torch.tensor(datasets_np).float(), torch.tensor(labels_np).float())

    # Step 4: Train model with K-Fold
    all_metrics, final_model = run_kfold_training(
        dataset=train_tensor,
        model_type=model_name,
        k_folds=10,
        batch_size=32,
        epochs=epochs,
        device=device,
        seed=seed
    )

    # Step 5: Save raw metrics as .npz
    np.savez(f"artifacts/metrics/training_metrics_{file_prefix}.npz", **{
        key: [v.numpy() for v in values]
        for key, values in all_metrics.items()
    })

    # Step 6: Compute and save averaged metrics
    averaged = {k: torch.stack(v).mean(dim=0) for k, v in all_metrics.items()}
    np.savez(f"artifacts/metrics/averaged_training_metrics_{file_prefix}.npz", **{
        key: value.numpy() for key, value in averaged.items()
    })

    # Step 7: Plot results
    plot_results(
        averaged["train_loss"], averaged["val_loss"],
        averaged["train_f1"], averaged["val_f1"],
        averaged["train_hamming"], averaged["val_hamming"],
        f"artifacts/plots/progression_{file_prefix}.pdf"
    )

    # Step 8: Save final model
    
    model_path = f"artifacts/models/model_{file_prefix}.pth"
    save_model(final_model, model_path)
    print(f"✅ Final model saved to: {model_path}")
