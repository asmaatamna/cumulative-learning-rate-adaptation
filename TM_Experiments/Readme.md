# Benchmarking ML Optimizers with MLCommons Algorithmic Efficiency

This project uses the [MLCommons Algorithmic Efficiency](https://github.com/mlcommons/algorithmic-efficiency) benchmark suite to evaluate and compare the performance of different optimization algorithms across a variety of supervised learning tasks.

## Purpose

The goal is to benchmark different optimizers—such as Adam, Adam with CLARA-based learning rate adaptation, and others—on standard tasks including:

- MNIST
- CIFAR-10
- ImageNet
- Wikitext-103
- And more...

## Structure

- `algorithmic-efficiency/`: The original benchmark framework cloned as a Git submodule.
- `optimizers/`: Custom implementations of optimizers to be plugged into the benchmark.
- `main.py`: Entry point to select tasks and optimizers, and run experiments.
- `experiments/`: Stores logs, checkpoints, and results per run.

## Conda Environment Setup

To ensure compatibility and reproducibility, we recommend setting up a clean Conda environment:

```bash
# Create and activate environment
conda create -n algoeff python=3.11 -y
conda activate algoeff

# Navigate to the benchmark repository root (where setup.py or pyproject.toml is located)
cd algorithmic-efficiency

# Install in editable mode with all dependencies
pip install -e '.[full]'
```

Note: You may have to edit the required version of some modules when they're not available
---
