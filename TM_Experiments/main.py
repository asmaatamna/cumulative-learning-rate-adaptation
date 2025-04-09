# ---------------------------------------------------------*\
# Title: Optimizer Benchmarking Suite - Main
# Author: TM 2025
# ---------------------------------------------------------*/

# Dataset & Model Loader
from src.datasets import load_dataset
from src.models import load_model

# Optimizer Setup
from src.optimizers import get_optimizer

# Training & Evaluation
from src.training import train_model
from src.testing import evaluate_model

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
DATASET_TO_BENCHMARK = "cifar10"    # Choose dataset for benchmarking

BATCH_SIZE = 128
NUM_CLASSES = 10

# 3. Training Parameters
# ---------------------------------------------------------*/
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42

# 4. Optimizers to Benchmark
# ---------------------------------------------------------*/
OPTIMIZERS = ["SGD", "SGDMomentum", "Adam", "AdamW", "RMSProp", "ADAM_CLARA"]

# 5. Save Paths
# ---------------------------------------------------------*/
SAVE_DIR = "./results/"
PLOT_DIR = "./results/plots/"

# ---------------------------------------------------------*/
# Run Functions
# ---------------------------------------------------------*/

def download_datasets():
    """Download and prepare all required datasets."""
    print("\n--------------------------------")
    print("Downloading and preparing datasets... 📚")
    print("--------------------------------")
    for dataset_name in DATASETS:
        try:
            _ = load_dataset(dataset_name, batch_size=BATCH_SIZE)
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
    print("All datasets are ready.")

def run_experiment():
    """Run the optimizer benchmarking experiments."""
    print("\n--------------------------------")
    print(f"Starting Benchmarking 🚀")
    print("--------------------------------")
    
    run_optimizer_benchmark(
        dataset_name="mnist",
        optimizers=["SGD", "Adam", "AdamW", "RMSProp", "Adam_Clara"],
        batch_size=128,
        num_classes=10,
        epochs=10,
        learning_rate=0.001,
        seed=42,
        save_dir="./results/"
    )


def plot_experiment_results():
    """Plot benchmarking results."""
    plot_results(save_dir=SAVE_DIR, plot_dir=PLOT_DIR)

# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/
if __name__ == "__main__":
    if DOWNLOAD_DATASETS:
        download_datasets()

    if RUN_BENCHMARK:
        run_experiment()

    if PLOT_RESULTS:
        plot_experiment_results()

    print("\n--------------------------------")
    print("All tasks completed. 🌟")
    print("--------------------------------")


#---------------------------------------------------------*/
#
#---------------------------------------------------------*/