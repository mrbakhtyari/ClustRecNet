import torch
import numpy as np
from sklearn.metrics import hamming_loss, f1_score
from torch.utils.data import DataLoader


def train_model(
    model_components: tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> tuple[
    list[float], list[float],
    list[float], list[float],
    list[float], list[float],
]:
    """
    Train and evaluate a PyTorch model over multiple epochs.

    Args:
        model_components (tuple): Tuple of (model, loss_fn, optimizer).
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of epochs to train.
        device (torch.device): Target device for training.

    Returns:
        Tuple containing:
            - train_loss, val_loss: list of loss values
            - train_f1, val_f1: list of micro F1 scores
            - train_hamming, val_hamming: list of Hamming losses
            - predictions from final test batch
    """
    model, loss_fn, optimizer = model_components
    model.to(device)

    # Initialize metric containers
    metric_names = ["loss", "hamming", "f1"]
    train_metrics = {name: [] for name in metric_names}
    val_metrics = {name: [] for name in metric_names}

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        batch_losses, batch_hammings, batch_f1s = [], [], []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds_binary = (torch.sigmoid(predictions) > 0.5).float()

            batch_losses.append(loss.item())
            batch_hammings.append(hamming_loss(y_batch.cpu().numpy(), preds_binary.cpu().numpy()))
            batch_f1s.append(f1_score(y_batch.cpu().numpy(), preds_binary.cpu().numpy(), average='micro'))

        train_metrics["loss"].append(np.mean(batch_losses))
        train_metrics["hamming"].append(np.mean(batch_hammings))
        train_metrics["f1"].append(np.mean(batch_f1s))

        # ---- Evaluation ----
        model.eval()
        val_losses, val_hammings, val_f1s = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch = x_batch.float()
                y_batch = y_batch.float()
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)

                preds_binary = (torch.sigmoid(predictions) > 0.5).float()

                val_losses.append(loss.item())
                val_hammings.append(hamming_loss(y_batch.cpu().numpy(), preds_binary.cpu().numpy()))
                val_f1s.append(f1_score(y_batch.cpu().numpy(), preds_binary.cpu().numpy(), average='micro'))

            val_metrics["loss"].append(np.mean(val_losses))
            val_metrics["hamming"].append(np.mean(val_hammings))
            val_metrics["f1"].append(np.mean(val_f1s))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss'][-1]:.4f}, Test Loss: {val_metrics['loss'][-1]:.4f} | "
            f"Train F1: {train_metrics['f1'][-1]:.4f}, Test F1: {val_metrics['f1'][-1]:.4f} | "
            f"Train Hamming: {train_metrics['hamming'][-1]:.4f}, Test Hamming: {val_metrics['hamming'][-1]:.4f}"
        )

        torch.cuda.empty_cache()


    return (
        train_metrics["loss"], val_metrics["loss"],
        train_metrics["f1"], val_metrics["f1"],
        train_metrics["hamming"], val_metrics["hamming"],

    )
