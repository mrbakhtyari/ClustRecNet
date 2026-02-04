import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List
from src.clust_utils.static import ALGORITHMS


def plot_results(train_loss, test_loss, train_f1, test_f1, train_hamming, test_hamming, save_path: Optional[str] = None):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    _plot_single_metric(ax[0], train_loss, test_loss, "Binary Cross-Entropy Loss", "(a)")
    _plot_single_metric(ax[1], train_f1, test_f1, "F1-score", "(b)")
    _plot_single_metric(ax[2], train_hamming, test_hamming, "Hamming distance", "(c)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def _plot_single_metric(ax, train, test, ylabel, title):
    ax.plot(train, 's-', label='Train')
    ax.plot(test, 'o-', label='Validation')
    ax.set_xlabel('Epochs', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=24)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)


def plot_data(dataset: np.ndarray, labels: np.ndarray, file_path: Optional[str] = None):
    plt.figure(figsize=(10, 8))
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels)
    plt.title("Clustered Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    if file_path:
        plt.savefig(f"{file_path}.pdf", dpi=300)
    plt.show()
    plt.close()


def plot_confusion_matrix(cm: List[np.ndarray], save_path: Optional[str] = None):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for i in range(len(cm)):
        ax = axes.flat[i]
        sns.heatmap(cm[i], annot=True, cmap='Reds', fmt='.4f', cbar=False, ax=ax,
                    xticklabels=['1', '0'], yticklabels=['1', '0'], annot_kws={'size': 15})
        ax.set_title(ALGORITHMS[i].display_name, fontsize=16)
        ax.set_xlabel('Predicted', fontsize=15)
        ax.set_ylabel('Actual', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_clustering_results(data: np.ndarray,
                            predictions: List[np.ndarray],
                            ari_scores: List[float],
                            save_path: Optional[str] = None):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten()

    for i, (pred, ari) in enumerate(zip(predictions, ari_scores)):
        ax = axes[i]
        ax.scatter(data[:, 0], data[:, 1], c=pred, cmap='viridis', s=10)
        ax.set_title(ALGORITHMS[i].display_name, fontsize=24)
        ax.set_xlabel('Feature 1', fontsize=18)
        ax.set_ylabel('Feature 2', fontsize=18)
        ax.text(0.05, 0.05, f'ARI: {ari:.2f}', transform=ax.transAxes, fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def algorithm_type_distribution(ari_binary: np.ndarray, save_path: Optional[str] = None):
    sum_columns = np.sum(ari_binary, axis=0).astype(int)
    labels = [alg.display_name for alg in ALGORITHMS]

    plt.bar(labels, sum_columns, align='center', alpha=0.5)
    plt.ylabel('Number of Samples')
    plt.xlabel('Algorithms')
    plt.xticks(rotation=90)
    plt.title("Number of Recommendations per Algorithm")
    for i, val in enumerate(sum_columns):
        plt.text(i, val, str(val), ha='center')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def recommendation_distribution(ari_binary: np.ndarray, save_path: Optional[str] = None):
    sum_rows = np.sum(ari_binary, axis=1).astype(int)
    unq, count = np.unique(sum_rows, return_counts=True)

    plt.bar(unq, count, align='center', alpha=0.5)
    plt.ylabel('Number of Samples')
    plt.xlabel('Number of Recommendations')
    plt.title("Distribution of Algorithm Recommendations")
    for i, val in enumerate(count):
        plt.text(unq[i], val, str(val), ha='center')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
