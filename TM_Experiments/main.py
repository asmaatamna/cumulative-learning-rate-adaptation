# ---------------------------------------------------------*\
# Title: Optimizer Benchmarking Suite - Main
# Author: TM 2025
# ---------------------------------------------------------*/

# Dataset & Model Loader
from src.datasets import load_dataset

# Benchmarking Utilities
from utils.benchmark import run_optimizer_benchmark, plot_results

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
DATASETS = ["mnist", "fmnist", "cifar10"]  # List of datasets to download
DATASET_TO_BENCHMARK = "mnist"             # Choose dataset for benchmarking

BATCH_SIZE = 128      # Batch size for training 
NUM_CLASSES = 10      # Number of classes in the dataset

# 3. Training Parameters
# ---------------------------------------------------------*/
EPOCHS = 20     
LEARNING_RATE = 0.00001
SEED = 42

# 4. Optimizers to Benchmark
# ---------------------------------------------------------*/
OPTIMIZERS = ["SGD", "SGDMomentum", "Adam", "AdamW", "RMSProp",
              "Adam_Clara_Global", "Adam_Clara_Local", "Adam_Clara_Smoothed"]

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
    print("All datasets are ready.")


def run_experiment(dataset_name, optimizers, batch_size, num_classes, epochs, learning_rate, seed, save_dir):
    """Run the optimizer benchmarking experiments."""
    print("\n--------------------------------")
    print(f"Starting Benchmarking 🚀")
    print("--------------------------------")

    run_optimizer_benchmark(
        dataset_name=dataset_name,
        optimizers=optimizers,
        batch_size=batch_size,
        num_classes=num_classes,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        save_dir=save_dir
    )


def plot_experiment_results(save_dir, plot_dir):
    """Plot benchmarking results."""
    plot_results(save_dir=save_dir, plot_dir=plot_dir)


# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/

if __name__ == "__main__":
    if DOWNLOAD_DATASETS:
        download_datasets(DATASETS, BATCH_SIZE)

    if RUN_BENCHMARK:
        run_experiment(
            dataset_name=DATASET_TO_BENCHMARK,
            optimizers=OPTIMIZERS,
            batch_size=BATCH_SIZE,
            num_classes=NUM_CLASSES,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            seed=SEED,
            save_dir=SAVE_DIR
        )

    if PLOT_RESULTS:
        plot_experiment_results(SAVE_DIR, PLOT_DIR)

    print("\n--------------------------------")
    print("All tasks completed. 🌟")
    print("--------------------------------")

# ---------------------------------------------------------*/
#
# ---------------------------------------------------------*/
