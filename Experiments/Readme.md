# Cumulative Learning Rate Adaptation: Revisiting Path-Based Schedules for SGD and Adam

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Introduction

This project explores cumulative learning rate adaptation techniques for optimization algorithms such as SGD and Adam. It revisits path-based schedules to improve convergence and performance in machine learning tasks. The project includes implementations of adaptive learning rate strategies, benchmarking experiments, and visualization of results.

## Installation

To set up the project locally, install the required dependencies:

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchdata datasets pyarrow matplotlib numpy
```

## Usage

1. **Download Datasets**: Run the script to download and prepare datasets.
   ```bash
   python Experiments/main.py --download-datasets
   ```

2. **Run Benchmarks**: Execute experiments to benchmark optimizers.
   ```bash
   python Experiments/main.py --run-benchmark
   ```

3. **Visualize Results**: Use the provided Jupyter notebooks to analyze and visualize optimization paths and metrics.

## Project Structure

```
.
├── Adam_CLARA_numerical_functions.ipynb   # Notebook for Adam + CLARA experiments
├── Generate-paper-plots.ipynb            # Notebook for generating plots for papers
├── Plot-learning-rate-experiments-results.ipynb  # Notebook for analyzing learning rate experiments
├── Experiments/
│   ├── main.py                           # Main script for running experiments
│   ├── Readme.md                         # Documentation for experiments
│   ├── data/                             # Dataset storage
│   ├── optimizers/                       # Optimizer implementations
│   ├── results/                          # Results and visualizations
│   ├── scripts/                          # Additional scripts
│   ├── src/                              # Source code
│   ├── utils/                            # Utility functions
│   └── 3D_Plotting.ipynb                 # Notebook for 3D visualizations
```

---