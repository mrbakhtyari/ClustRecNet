# ClustRecNet: A Novel End-to-End Deep Learning Framework for Clustering Algorithm Recommendation

**ClustRecNet** is a deep learning framework that recommends the most suitable clustering algorithm for a given dataset by directly learning high-order topological representations. Unlike traditional approaches that rely on handcrafted meta-features, ClustRecNet eliminates the knowledge bottleneck by learning structural patterns directly from raw data using a hybrid architecture combining CNN, ResNet, and attention mechanisms.

Trained on a repository of **34,000 synthetic datasets** with diverse structures, ClustRecNet significantly outperforms traditional Cluster Validity Indices (CVIs) such as Silhouette, Calinski-Harabasz, Davies-Bouldin, and Dunn, as well as state-of-the-art AutoML frameworks including ML2DAC, AutoML4Clust, and AutoCluster.

## ✨ Key Features

- **End-to-End Learning**: Fully trainable deep neural network that learns latent structural representations directly from input data distributions.
- **No Meta-Feature Engineering**: Eliminates the information bottleneck caused by manual feature extraction, allowing the model to capture high-order structural patterns.
- **Hybrid Architecture**: Integrates CNN, Residual blocks, and Attention mechanisms to capture both local spatial features and global contextual dependencies.
- **Large-Scale Training**: Built on 34,000 diverse synthetic datasets with controlled variation across clusters, objects, features, and noise levels.
- **State-of-the-Art Performance**: Achieves a **0.497 ARI gain** over Calinski-Harabasz index on synthetic data and **44.16% ARI improvement** over the leading AutoML approach (ML2DAC) on real-world benchmarks.

## 🏗️ Architecture Overview

ClustRecNet integrates four principal blocks:

1. **CNN Block**: 2D convolutional layer with batch normalization and max pooling for initial feature extraction.
2. **Residual Blocks (×2)**: Two consecutive residual blocks with skip connections to extract hierarchical features and address vanishing gradient issues.
3. **Self-Attention Mechanism**: Captures long-range dependencies within feature maps, complementing the local processing of convolutional layers.
4. **Fully Connected Layers**: Final classification layers producing logits for 10 clustering algorithms.

## 📁 Repository Structure

```
ClustRecNet/
├── configs/             # Configuration files for different model variants
├── data/                # Real-world and synthetic datasets
├── models/              # Trained model checkpoints
├── notebooks/           # Jupyter notebooks for evaluation and visualization
├── results/             # Output metrics and predictions
├── src/                 # Core source code
│   ├── analysis/        # Evaluation metrics and ablation study
│   ├── clust_utils/     # Clustering utilities and helpers
│   ├── clustering/      # Clustering algorithm implementations
│   ├── data_generation/ # Synthetic data generation pipeline
│   └── training/        # Model definitions and training loops
├── tests/               # tests
├── main.py              # Entry point for training
├── ablation.py          # Run ablation studies on real datasets
├── pyproject.toml       # Project configuration (for uv)
└── requirements.txt     # Project dependencies (for pip)
```

## 🚀 Installation

### Option 1: Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrbakhtyari/ClustRecNet.git
cd ClustRecNet

# Create virtual environment and install dependencies
uv sync
```

### Option 2: Using `pip`

```bash
# Clone the repository
git clone https://github.com/mrbakhtyari/ClustRecNet.git
cd ClustRecNet

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Running the Pipeline

### Train the Model

Train on synthetic data with a selected model variant:

```bash
# Using uv
uv run python main.py --data-path="data/" --model-name="MODEL_NAME"

# Using pip (with activated venv)
python main.py --data-path="data/" --model-name="MODEL_NAME"
```

**Available `--model-name` options:**

| Model Name     | Description                        |
|----------------|------------------------------------|
| `resnet`       | Full CNN + ResNet + ATN (default)  |
| `baseline_cnn` | Baseline CNN only                  |
| `no_cnn`       | ResNet + ATN (ablation)            |
| `no_res`       | CNN + ATN (ablation)               |
| `no_att`       | CNN + ResNet (ablation)            |

### Run Ablation Analysis

Evaluate model performance across real-world datasets:

```bash
# Using uv
uv run python ablation.py \
  --data-path="data/real_world_datasets" \
  --model-path="models/MODEL_NAME.pth" \
  --model-name="MODEL_NAME"

# Using pip (with activated venv)
python ablation.py \
  --data-path="data/real_world_datasets" \
  --model-path="models/MODEL_NAME.pth" \
  --model-name="MODEL_NAME"
```

## 📊 Clustering Algorithms Evaluated

ClustRecNet evaluates and recommends from 10 widely-used clustering algorithms:

| Category         | Algorithms                                      |
|------------------|------------------------------------------------|
| **Partitioning** | k-means, k-medians                             |
| **Hierarchical** | Agglomerative Clustering, BIRCH, Ward's method |
| **Density-based**| DBSCAN, HDBSCAN, OPTICS                        |
| **Probabilistic**| Gaussian Mixture Models (GMM)                  |
| **Graph-based**  | Spectral Clustering                            |

## 📈 Results Summary

### Synthetic Data (Test Set)

| Method          | F1-score (↑)    | Hamming (↓)     | ARI (↑)         |
|-----------------|-----------------|-----------------|-----------------|
| **ClustRecNet** | **0.757 ± 0.017** | **0.183 ± 0.011** | **0.878 ± 0.007** |
| Silhouette      | 0.720 ± 0.009   | 0.280 ± 0.009   | 0.365 ± 0.020   |
| Calinski-Harabasz | 0.669 ± 0.013 | 0.331 ± 0.013   | 0.381 ± 0.017   |
| Davies-Bouldin  | 0.681 ± 0.014   | 0.319 ± 0.014   | 0.344 ± 0.016   |
| Dunn            | 0.637 ± 0.011   | 0.363 ± 0.011   | 0.280 ± 0.017   |

### Real-World Datasets (UCI Benchmarks)

| Model                | Median ARI | Mean ARI   |
|----------------------|------------|------------|
| **ClustRecNet (CH)** | **0.1794** | **0.2520** |
| **ClustRecNet (Sil)**| **0.1684** | **0.2266** |
| Baseline CNN         | 0.1119     | 0.2101     |
| AutoCluster (CH)     | 0.1009     | 0.1474     |
| AML4C                | 0.0730     | 0.1412     |
| ML2DAC               | 0.1305     | 0.1748     |

ClustRecNet demonstrates statistically significant improvements (Wilcoxon signed-rank test, p < 0.05) over all competing methods.

## 🖥️ Hardware Requirements

To reproduce the exact results:

```
Torch version: 2.6.0
Compiled with CUDA: 12.2
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
```

Training proceeds for 30 epochs with a batch size of 32 using 10-fold cross-validation.

## 📄 Citation

If you use ClustRecNet in your research, please cite:

**BibTeX**
```bibtex
@ARTICLE{11535743,
  author={Bakhtyari, Mohammadreza and Mazoure, Bogdan and Amorim, Renato Cordeiro De and Rabusseau, Guillaume and Makarenkov, Vladimir},
  journal={IEEE Access},
  title={ClustRecNet: A Novel End-to-End Deep Learning Framework for Clustering Algorithm Recommendation},
  year={2026},
  volume={14},
  pages={81352-81365},
  doi={10.1109/ACCESS.2026.3697689}
}
```
