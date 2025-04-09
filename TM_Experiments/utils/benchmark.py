# ---------------------------------------------------------*\
# Title: Optimizer Benchmark Runner
# Author: TM
# ---------------------------------------------------------*/

import os
import torch
from src.datasets import load_dataset
from src.models import load_model
from src.optimizers import get_optimizer
from src.training import train_model
from src.testing import evaluate_model
import pickle
import matplotlib.pyplot as plt

def run_optimizer_benchmark(dataset_name, optimizers, batch_size, num_classes, epochs, learning_rate, seed, save_dir):
    """Runs benchmarking for different optimizers on a given dataset.

    Args:
        dataset_name (str): Dataset to use ("mnist", "fmnist", "cifar10").
        optimizers (list): List of optimizer names.
        batch_size (int): Batch size for training/testing.
        num_classes (int): Number of classes.
        epochs (int): Number of training epochs.
        learning_rate (float): Base learning rate.
        seed (int): Random seed for reproducibility.
        save_dir (str): Directory to save benchmark results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Load data
    train_loader, test_loader = load_dataset(dataset_name, batch_size=batch_size)

    results = {}

    for optimizer_name in optimizers:
        print(f"\n=== Running Benchmark: {optimizer_name.upper()} on {dataset_name.upper()} ===")

        # Load fresh model
        model = load_model(dataset_name, num_classes=num_classes)

        # Load optimizer
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate=learning_rate)

        # Train model
        train_losses, train_accuracies = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device
        )

        # Evaluate model
        test_loss, test_accuracy = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )

        # Save results
        results[optimizer_name] = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }

    # Save all benchmark results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_file = os.path.join(save_dir, f"benchmark_{dataset_name}.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✅ Benchmarking completed. Results saved to {results_file}")

# ---------------------------------------------------------*\
# Benchmark Results Plotting
# ---------------------------------------------------------*/

def plot_results(save_dir="./results/", plot_dir="./results/plots/"):
    """Loads benchmark results and generates plots.

    Args:
        save_dir (str): Directory where benchmark .pkl files are stored.
        plot_dir (str): Directory to save plots.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Find all result files
    result_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    for result_file in result_files:
        dataset_name = result_file.replace("benchmark_", "").replace(".pkl", "")

        # Load results
        with open(os.path.join(save_dir, result_file), "rb") as f:
            results = pickle.load(f)

        # Prepare plots
        optimizers = list(results.keys())

        # ---------------- Training Loss Plot ----------------
        plt.figure(figsize=(10, 6))
        for opt_name in optimizers:
            plt.plot(results[opt_name]["train_losses"], label=opt_name)
        plt.title(f"Training Loss - {dataset_name.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_train_loss.png"), dpi=300)
        plt.show()
        plt.close()

        # ---------------- Training Accuracy Plot ----------------
        plt.figure(figsize=(10, 6))
        for opt_name in optimizers:
            plt.plot(results[opt_name]["train_accuracies"], label=opt_name)
        plt.title(f"Training Accuracy - {dataset_name.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_train_accuracy.png"), dpi=300)
        plt.show()
        plt.close()

        # ---------------- Final Test Accuracy Bar Plot ----------------
        test_accuracies = [results[opt]["test_accuracy"] for opt in optimizers]

        plt.figure(figsize=(8, 6))
        plt.bar(optimizers, test_accuracies)
        plt.title(f"Final Test Accuracy - {dataset_name.upper()}")
        plt.xlabel("Optimizer")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_test_accuracy.png"), dpi=300)
        plt.show()
        plt.close()

        print(f"✅ Plots saved for {dataset_name.upper()} in {plot_dir}")

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\