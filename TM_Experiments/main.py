# ---------------------------------------------------------*\
# Title: Optimizer Benchmarking Suite - Main
# Author: TM 2025
# ---------------------------------------------------------*/

import os

# Dataset & Model Loader
from src.datasets import load_dataset

# Benchmarking Utilities
from utils.benchmark import run_optimizer_benchmark, plot_results, plot_results_for_dataset

# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/

# 1. Experiment Settings
# ---------------------------------------------------------*/
DOWNLOAD_DATASETS = 1   # Download and prepare datasets
RUN_BENCHMARK = 1       # Run optimizer benchmark
PLOT_RESULTS = 1        # Generate result plots

# 2. Dataset Parameters
# ---------------------------------------------------------*/
# DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "breast_cancer", "wikitext"]  # All datasets you prepared
DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "breast_cancer"]

SUBSET = 5            # Percentage of dataset to use (use full)

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
EPOCHS = 2             # More realistic number of epochs
SEED = 42

# 4. Optimizers to Benchmark
# ---------------------------------------------------------*/
OPTIMIZERS = ["SGD", "SGDMomentum", "Adam", "AdamW", "RMSProp",
              "Adam_Clara_Global", "Adam_Clara_Local", "Adam_Clara_Smoothed"]

# OPTIMIZERS = ["SGD", "Adam",
#               "Adam_Clara_Global", "Adam_Clara_Local", "Adam_Clara_Smoothed"]

# Default Learning Rates per Optimizer
OPTIMIZER_LR = {
    "SGD": 0.0000001,
    "Adam": 0.0000001,
    "Adam_Clara_Global": 0.0000001,
    "Adam_Clara_Local": 0.0000001,
    "Adam_Clara_Smoothed": 0.0000001,
}

# 5. Save Paths
# ---------------------------------------------------------*/
SAVE_DIR = "./results/"
PLOT_DIR = "./results/plots/"

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

        for dataset in DATASETS:
            if dataset not in NUM_CLASSES_DICT:
                raise ValueError(f"Dataset {dataset} not found in NUM_CLASSES_DICT!")

            dataset_result_dir = os.path.join(SAVE_DIR, dataset)
            os.makedirs(dataset_result_dir, exist_ok=True)

            if dataset in ["wikitext", "bookcorpus"]:
                batch_size = 8  # Much smaller for transformers
            else:
                batch_size = BATCH_SIZE
    
            for optimizer in OPTIMIZERS:
                learning_rate = OPTIMIZER_LR.get(optimizer, 0.001)

                run_optimizer_benchmark(
                    dataset_name=dataset,
                    optimizers=[optimizer],
                    batch_size=BATCH_SIZE,
                    num_classes=NUM_CLASSES_DICT[dataset],
                    epochs=EPOCHS,
                    learning_rate=learning_rate,
                    seed=SEED,
                    save_dir=dataset_result_dir,  # ⚡ Save in dataset-specific folder
                    subset=SUBSET,
                )

            # ⚡ After all optimizers finished -> Plot all results for this dataset
            plot_results_for_dataset(dataset_result_dir)

        print("\n--------------------------------")
        print("All tasks completed. 🌟")
        print("--------------------------------")

    if PLOT_RESULTS:
        plot_results(SAVE_DIR, PLOT_DIR)

    print("\n--------------------------------")
    print("All tasks completed. 🌟")
    print("--------------------------------")

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
