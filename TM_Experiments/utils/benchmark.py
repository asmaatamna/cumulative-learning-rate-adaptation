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
from tabulate import tabulate


def run_optimizer_benchmark(dataset_name, optimizers, batch_size, num_classes, epochs, learning_rate, seed, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nRunning benchmark on {device.upper()} with dataset: {dataset_name.upper()}")

    # Setze Seed
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Lade Daten
    train_loader, test_loader = load_dataset(dataset_name, batch_size=batch_size)
    results = {}

    for optimizer_name in optimizers:
        print(f"\n=== Running Benchmark: {optimizer_name.upper()} on {dataset_name.upper()} ===")
        model = load_model(dataset_name, num_classes=num_classes)
        model.to(device)
        
        # Initialevaluation vor Training, falls erwünscht (optional)
        init_loss, init_accuracy = evaluate_model(model, test_loader, device=device)
        train_losses = [init_loss]
        train_accuracies = [init_accuracy]
        lr_history = [learning_rate]

        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate=learning_rate)
        
        t_losses, t_accuracies, t_lr = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=device
        )
        train_losses.extend(t_losses)
        train_accuracies.extend(t_accuracies)
        lr_history.extend(t_lr)

        test_loss, test_accuracy = evaluate_model(model, test_loader, device=device)

        results[optimizer_name] = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "lr_history": lr_history,   # Lernrate pro Epoche
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }

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
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    result_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    for result_file in result_files:
        dataset_name = result_file.replace("benchmark_", "").replace(".pkl", "")
        with open(os.path.join(save_dir, result_file), "rb") as f:
            results = pickle.load(f)

        optimizers = list(results.keys())

        # Plot: Training Loss
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

        # Plot: Training Accuracy
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

        # Plot: Final Test Accuracy (Bar Plot)
        test_accuracies = [results[opt]["test_accuracy"] for opt in optimizers]
        plt.figure(figsize=(8, 6))
        plt.bar(optimizers, test_accuracies)
        plt.title(f"Final Test Accuracy - {dataset_name.upper()}")
        plt.xlabel("Optimizer")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_test_accuracy.png"), dpi=300)
        plt.show()
        plt.close()

        # Plot: Learning Rate Development per Epoch
        plt.figure(figsize=(10, 6))
        for opt_name in optimizers:
            plt.plot(results[opt_name]["lr_history"], label=opt_name)
        plt.title(f"Learning Rate Development - {dataset_name.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_learning_rate.png"), dpi=300)
        plt.show()
        plt.close()
        print(f"✅ Plots saved for {dataset_name.upper()} in {plot_dir}")
        output_lr_table(save_dir=save_dir)


def output_lr_table(save_dir="./results/"):
    """Load benchmark results and export a table of learning rates per optimizer."""
    result_files = [f for f in os.listdir(save_dir) if f.startswith("benchmark_") and f.endswith(".pkl")]

    for result_file in result_files:
        dataset_name = result_file.replace("benchmark_", "").replace(".pkl", "")

        with open(os.path.join(save_dir, result_file), "rb") as f:
            results = pickle.load(f)

        headers = ["Optimizer", "Initial LR", "Final LR", "Min LR", "Max LR", "Mean LR"]
        table = []

        for opt_name, res in results.items():
            lr_hist = res.get("lr_history", [])
            if lr_hist:
                initial_lr = lr_hist[0]
                final_lr = lr_hist[-1]
                min_lr = min(lr_hist)
                max_lr = max(lr_hist)
                mean_lr = sum(lr_hist) / len(lr_hist)
            else:
                initial_lr = final_lr = min_lr = max_lr = mean_lr = None
            table.append([opt_name, initial_lr, final_lr, min_lr, max_lr, mean_lr])

        table_str = tabulate(table, headers=headers, tablefmt="grid")

        output_file = os.path.join(save_dir, f"{dataset_name}_lr_table.txt")
        with open(output_file, "w") as f:
            f.write(table_str)

        print(f"✅ Lernraten-Tabelle für {dataset_name.upper()} gespeichert in {output_file}")



#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\