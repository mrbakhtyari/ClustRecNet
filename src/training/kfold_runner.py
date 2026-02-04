import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from src.training.models import build_model
from src.training.training_loop import train_model


def run_kfold_training(
    dataset,
    model_type: str = "res_net",
    k_folds: int = 10,
    batch_size: int = 32,
    epochs: int = 30,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed: int = 0
) -> tuple[dict, object]:
    """
    Train model using k-fold cross-validation.

    Returns:
        - metrics_dict: dictionary of all train/test metrics
        - final_model: last model object trained (used for saving)
    """
    metric_names = ["loss", "f1", "hamming"]
    all_metrics = {f"train_{m}": [] for m in metric_names} | {f"val_{m}": [] for m in metric_names}

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    final_model = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n==== Fold {fold + 1} ====")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
        model = build_model(model_type=model_type)

        train_loss, val_loss, train_f1, val_f1, train_hamming, val_hamming = train_model(
            model, train_loader, val_loader, epochs, device
        )

        all_metrics["train_loss"].append(torch.tensor(train_loss, dtype=torch.float32))
        all_metrics["val_loss"].append(torch.tensor(val_loss, dtype=torch.float32))
        all_metrics["train_f1"].append(torch.tensor(train_f1, dtype=torch.float32))
        all_metrics["val_f1"].append(torch.tensor(val_f1, dtype=torch.float32))
        all_metrics["train_hamming"].append(torch.tensor(train_hamming, dtype=torch.float32))
        all_metrics["val_hamming"].append(torch.tensor(val_hamming, dtype=torch.float32))

        final_model = model  # Save last model for later

    return all_metrics, final_model