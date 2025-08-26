# ---------------------------------------------------------*\
# Title: Optimizer Benchmarking Suite - Main
# Author: TM 2025
# ---------------------------------------------------------*/

import os
import argparse

# Parallelize training loop (1 process per initial lr value)
from multiprocessing import Process

# Dataset & Model Loader
from src.datasets import load_dataset

# Benchmarking Utilities
from utils.benchmark import run_for_lr
from datetime import datetime

import wandb


# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/

# 1. Experiment Settings
# ---------------------------------------------------------*/
DOWNLOAD_DATASETS = 0   # Download and prepare datasets
RUN_BENCHMARK = 1       # Run optimizer benchmark
PLOT_RESULTS = 1        # Generate result plots

EXPERIMENT_NAME = "Experiment 11"  # Name of the experiment

# 2. Dataset Parameters
# ---------------------------------------------------------*/
DATASETS = ["breast_cancer", "iris", "wine", "digits", "mnist", "fmnist", "cifar10", "cifar100"]

SUBSET = 100            # Percentage of dataset to use (use full)

BATCH_SIZE = 128        # Batch size for training (adapted: 128 better for transformers too)

# Dataset to Number of Classes
NUM_CLASSES_DICT = {
    "mnist": 10,
    "fmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "breast_cancer": 2,
    "iris": 3,
    "wine": 3,
    "digits": 10,
    "wikitext": 50257  # For distilgpt2 tokenizer (vocab size)
}

# 3. Training Parameters
# ---------------------------------------------------------*/
EPOCHS = 100
SEEDS = [0, 1, 2, 3, 4]

# 4. Optimizers to Benchmark
OPTIMIZERS = ["SGD_CLARA", "SGD_CLARA_us", "SGD", "Adam_CLARA", "Adam_CLARA_us", "Adam", "D-Adaptation"]

# Set a default learning rate for all optimizers
DEFAULT_LR = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# Set default damping values for CLARA algorithms (based on tuning done in a separate procedure)
d = 1e-3
DAMPING = {
    optimizer: {
        dataset: {
            str(lr): d  # Different damping values can be used for different (dataset, lr0) combinations
            for lr in DEFAULT_LR
        }
        for dataset in DATASETS
    }
    for optimizer in OPTIMIZERS
}

# 5. Save Paths
# ---------------------------------------------------------*/
SAVE_DIR = "./results/"

# ---------------------------------------------------------*/
# Run Functions
# ---------------------------------------------------------*/


def download_datasets(datasets, batch_size):
    """Download and prepare all required datasets."""
    print("\n--------------------------------")
    print("Downloading and preparing datasets... 📚")
    print("--------------------------------")
    for dataset_name in datasets:
        try:
            _ = load_dataset(dataset_name, batch_size=batch_size)
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
    print("All datasets are ready. ⛳️")

# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/


if __name__ == "__main__":
    if DOWNLOAD_DATASETS:
        download_datasets(DATASETS, batch_size=BATCH_SIZE)

    if RUN_BENCHMARK:
        print("\n--------------------------------")
        print(f"Starting Benchmarking 🚀")
        print("--------------------------------")

        for seed in SEEDS:
            # TODO: Add message displaying seed number

            # 📅 Build timestamped result directory name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            # run_folder = f"{timestamp}_{EPOCHS}_{seed}_{d:.0e}_DateTimeEpochSeedDamping"  # Use only in damping experiments
            run_folder = f"{timestamp}_{EPOCHS}_{seed}_DateTimeEpochSeed"
            save_path_with_time = os.path.join(SAVE_DIR, run_folder)

            for dataset in DATASETS:
                if dataset not in NUM_CLASSES_DICT:
                    raise ValueError(f"Dataset {dataset} not found in NUM_CLASSES_DICT!")

                batch_size = 8 if dataset in ["wikitext", "bookcorpus"] else BATCH_SIZE  # Much smaller for transformers

                # Support both single value and list for DEFAULT_LR
                learning_rates = DEFAULT_LR if isinstance(DEFAULT_LR, list) else [DEFAULT_LR]

                processes = []

                for lr in learning_rates:
                    # Add learning rate to subfolder name
                    lr_str = f"lr{lr:.0e}" if lr < 1 else f"lr{lr:.2f}"

                    # 🗂️ Create dataset-specific subfolder in timestamped folder
                    dataset_result_dir = os.path.join(save_path_with_time, dataset, lr_str)
                    os.makedirs(dataset_result_dir, exist_ok=True)

                    p = Process(
                        target=run_for_lr,
                        args=(
                            lr,
                            DAMPING,
                            dataset_result_dir,
                            dataset,
                            batch_size,
                            NUM_CLASSES_DICT[dataset],
                            OPTIMIZERS,
                            EPOCHS,
                            seed,
                            SUBSET,
                            EXPERIMENT_NAME
                        )
                    )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

        
    print("\n--------------------------------")
    print("All tasks completed. 🌟")
    print("--------------------------------")

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
