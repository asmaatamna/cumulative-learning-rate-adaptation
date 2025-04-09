# Benchmarking ML Optimizers with DeepOBS

This project uses the [DeepOBS](https://github.com/fsschneider/DeepOBS) benchmark suite to evaluate and compare the performance of different optimization algorithms across a variety of supervised learning tasks.

## Purpose

The goal is to benchmark different optimizers—such as Adam, SGD, RMSProp, and others—on standard tasks including:

- MNIST
- CIFAR-10
- Fashion-MNIST
- LSTM-based text datasets
- And more...

## Structure

- `deepobs/`: The original DeepOBS benchmark framework cloned as a Git submodule.
- `optimizers/`: Custom implementations of optimizers to be plugged into DeepOBS.
- `main.py`: Entry point to select tasks and optimizers, and run experiments.
- `experiments/`: Stores logs, checkpoints, and results per run.

## Cloning the Repository

To get started, clone the repository and its submodules:

```bash
cd TM_Experiments
# Clone the repository
git clone --branch develop --depth 1 https://github.com/fsschneider/DeepOBS.git

# Navigate into the repository
cd DeepOBS
```

## Conda Environment Setup

To ensure compatibility and reproducibility, we recommend setting up a clean Conda environment:

```bash
# Create and activate environment
conda config --add channels defaults
conda activate deepobs
```

## Installation

Install DeepOBS directly from PyPI:

```bash
pip install deepobs
```

## Downloading Benchmark Datasets

DeepOBS requires benchmark datasets to be available locally. These can be automatically downloaded using the `deepobs_prepare_data` utility:

```bash
./deepobs/scripts/deepobs_prepare_data.py 
```

Datasets are stored by default in a folder called `.deepobs` in the home directory.
For more options, refer to the [DeepOBS Documentation](https://deepobs.readthedocs.io/en/stable/api/scripts/deepobs_prepare_data.html).

## Running Experiments

To run a benchmark experiment with a specific optimizer and task:

```bash
python main.py --optimizer <optimizer_name> --task <task_name> --backend <tensorflow|pytorch>
```

### Example

```bash
python main.py --optimizer adam --task cifar10_3c3d --backend pytorch
```

### Command Line Arguments

- `--optimizer`: Name of the optimizer to benchmark (must match the implementation in `optimizers/`).
- `--task`: Name of the benchmark task, e.g., `mnist_mlp`, `cifar10_3c3d`.
- `--backend`: Deep learning framework to use (`tensorflow` or `pytorch`).

### Output

- Logs, checkpoints, and performance metrics will be saved in the `experiments/` directory.
- Training curves and optimizer performance can be visualized using the logs.

---

**Hinweise**:  
- Jetzt wird korrekt `pip install deepobs` verwendet.
- Die Anweisungen zum Download der Datensätze via `deepobs_prepare_data` sind vollständig integriert.
- Der Bezug auf die Original-Dokumentation ist verlinkt.

Möchtest du noch ein Beispiel-Output hinzufügen (z.B. typische Logdateien oder eine Beispiel-Plot-Ausgabe)?  
Das wäre oft hilfreich für neue Nutzer.