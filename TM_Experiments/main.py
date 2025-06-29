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

EXPERIMENT_NAME = "Experiment 3"  # Name of the experiment

# 2. Dataset Parameters
# ---------------------------------------------------------*/
# DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "breast_cancer", "wikitext"]  # All datasets you prepared
# DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "breast_cancer"]
DATASETS = ["mnist"]
# DATASETS = ["cifar10"]
# DATASETS = ["mnist"]

SUBSET = 100            # Percentage of dataset to use (use full)

BATCH_SIZE = 128        # Batch size for training (adapted: 128 better for transformers too)

# Dataset to Number of Classes
NUM_CLASSES_DICT = {
    "mnist": 10,
    "fmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "breast_cancer": 2,
    "wikitext": 50257  # For distilgpt2 tokenizer (vocab size)
}

# 3. Training Parameters
# ---------------------------------------------------------*/
EPOCHS = 5
SEEDS = [42]

# 4. Optimizers to Benchmark
# ---------------------------------------------------------*/
# OPTIMIZERS = ["SGD", "SGDMomentum", "Adam", "AdamW", "RMSProp",
#               "Adam_Clara_Global", "Adam_Clara_Local", "Adam_Clara_Smoothed", "SGD_CLARA"]
# OPTIMIZERS = ["SGD", "Adam", "AdamW", "SGD_CLARA",  "AdamW_CLARA"]
# OPTIMIZERS = ["SGD_CLARA", "Adam_CLARA", "Adam", "AdamW"]
# OPTIMIZERS = ["SGD", "SGD_CLARA", "Adam", "Adam_CLARA"]
OPTIMIZERS = ["SGD_CLARA", "SGD_CLARA_us", "SGD", "Adam_CLARA", "Adam_CLARA_us", "Adam"]  # TODO: Add D-Adaptation


# Set a default learning rate for all optimizers
DEFAULT_LR = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

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

    # Create parser
    parser = argparse.ArgumentParser(description="Run optimizer benchmark.")

    # Add damping argument
    parser.add_argument(
        "-d", "--damping",
        type=float,
        default=1e-3,
        help="Damping factor (default: 1e-3)"
    )

    # Parse arguments
    args = parser.parse_args()

    DAMPING = args.damping  # Damping parameter d in CLARA. TODO: Try values: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
    print("Damping factor:", DAMPING)
    
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
            run_folder = f"{timestamp}_{EPOCHS}_{seed}_{DAMPING:.0e}_DateTimeEpochSeedDamping"  # TODO: Add damping and seed
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
