# Adversarial Optimizer Study

This project investigates the impact of different optimizers on the performance and robustness of adversarially trained neural networks using PGD-based training.

## 🔍 Project Goals

- Compare SGD, Adam, AdamW, RMSProp, and AdaBelief in adversarial training.
- Evaluate clean accuracy, adversarial robustness, convergence speed, and training stability.
- Provide reproducible code and benchmark results.

## 🧪 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📦 Directory Structure

```
configs/        # YAML config files for each optimizer/dataset
models/         # Model architectures (e.g. ResNet)
train/          # Training and PGD attack code
eval/           # Evaluation scripts
utils/          # Logging, plotting, and helpers
results/        # Training logs, plots, metrics
notebooks/      # Jupyter notebooks for analysis
```
