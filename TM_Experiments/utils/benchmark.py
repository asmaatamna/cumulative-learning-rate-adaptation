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
import time


#---------------------------------------------------------*/
# Parameters
#---------------------------------------------------------*/
OPTIMIZER_COLORS = {
    "SGD": "blue",
    "SGDMomentum": "cyan",
    "Adam": "green",
    "AdamW": "darkgreen",
    "RMSProp": "brown",
    "Adam_Clara_Global": "#FF6347",     # Tomato 
    "Adam_Clara_Local": "#FF4500",      # OrangeRed
    "Adam_Clara_Smoothed": "#DC143C",   # Crimson
}

# Fallback if a color is missing
DEFAULT_COLOR = "gray"

def get_color(optimizer_name):
    return OPTIMIZER_COLORS.get(optimizer_name, DEFAULT_COLOR)

#---------------------------------------------------------*/
# Benchmarking
#---------------------------------------------------------*/

def run_optimizer_benchmark(dataset_name, optimizers, batch_size, num_classes, epochs, learning_rate, seed, save_dir, subset=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning benchmark on {device.upper()} with dataset: {dataset_name.upper()}")
    start_time = time.time()

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # ----------------------------------------
    # Load dataset
    # ----------------------------------------
    train_loader, test_loader = load_dataset(dataset_name, batch_size=batch_size, subset_percent=subset)

    # ----------------------------------------
    # Special handling for tabular input_dim
    # ----------------------------------------
    input_dim = None
    if dataset_name in ["breast_cancer", "iris", "wine", "digits"]:
        first_batch = next(iter(train_loader))
        inputs, _ = first_batch
        input_dim = inputs.shape[1]

    # ----------------------------------------
    # Check if Language Dataset
    # ----------------------------------------
    is_language_model = dataset_name in ["tinystories", "wikitext", "language_toy", "bookcorpus"]

    # ----------------------------------------
    # Loop over optimizers
    # ----------------------------------------
    for optimizer_name in optimizers:
        # Choose correct model
        if is_language_model:
            model = load_model(dataset_name, num_classes=num_classes, input_dim=input_dim, model_type="tinytransformer")
        else:
            model = load_model(dataset_name, num_classes=num_classes, input_dim=input_dim)

        model.to(device)

        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate=learning_rate)

        # ⚡ Pass is_language_model here!
        init_loss, init_accuracy = evaluate_model(
            model, 
            test_loader, 
            device=device, 
            is_language_model=is_language_model  # <-- HIER
        )

        train_losses = [init_loss]
        train_accuracies = [init_accuracy]
        lr_history = [optimizer.param_groups[0]['lr']]

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

        # ⚡ Again for final evaluation
        test_loss, test_accuracy = evaluate_model(
            model, 
            test_loader, 
            device=device, 
            is_language_model=is_language_model 
        )

        results = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "lr_history": lr_history,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }

        result_file = os.path.join(save_dir, f"{optimizer_name}.pkl")
        with open(result_file, "wb") as f:
            pickle.dump(results, f)

        print(f"✅ Saved benchmark for {optimizer_name.upper()} to {result_file}")

    elapsed_time = time.time() - start_time
    print(f"⏱️ Benchmark for {dataset_name.upper()} completed in {int(elapsed_time // 60)} min {elapsed_time % 60:.2f} sec")

# ---------------------------------------------------------*/
# Plot results from general results folder
# ---------------------------------------------------------*/
def plot_results(save_dir="./results/", plot_dir="./results/"):
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    result_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    for result_file in result_files:
        dataset_name = result_file.replace("benchmark_", "").replace(".pkl", "")

        with open(os.path.join(save_dir, result_file), "rb") as f:
            results = pickle.load(f)

        optimizers = list(results.keys())

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Benchmark Results - {dataset_name.upper()}", fontsize=20)

        # ---------------- Final Test Accuracy
        ax = axs[0, 0]
        test_accuracies = [results[opt]["test_accuracy"] for opt in optimizers]
        sorted_data = sorted(zip(test_accuracies, optimizers))
        sorted_accuracies, sorted_optimizers = zip(*sorted_data)
        
        colors = [get_color(opt) for opt in sorted_optimizers]

        ax.bar(sorted_optimizers, sorted_accuracies, color=colors)
        ax.set_title("Final Test Accuracy")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(range(len(sorted_optimizers)))
        ax.set_xticklabels(sorted_optimizers, rotation=45)
        ax.grid(axis="y")

        # ---------------- Training Accuracy
        ax = axs[0, 1]
        for opt_name in optimizers:
            ax.plot(results[opt_name]["train_accuracies"], label=opt_name, color=get_color(opt_name))
        ax.set_title("Training Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid()
        ax.set_xticks(range(len(results[optimizers[0]]["train_accuracies"])))  # Nur ganzzahlige Labels
        
        # ---------------- Training Loss
        ax = axs[1, 0]
        for opt_name in optimizers:
            ax.plot(results[opt_name]["train_losses"], label=opt_name, color=get_color(opt_name))
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()
        ax.set_xticks(range(len(results[optimizers[0]]["train_losses"])))  # Nur ganzzahlige Labels
        
        # ---------------- Learning Rate
        ax = axs[1, 1]
        for opt_name in optimizers:
            ax.plot(results[opt_name]["lr_history"], label=opt_name, color=get_color(opt_name))
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.legend()
        ax.grid()
        ax.set_xticks(range(len(results[optimizers[0]]["lr_history"])))  # Nur ganzzahlige Labels

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        base_path_png = os.path.join(plot_dir, f"{dataset_name}_full_benchmark.png")
        base_path_svg = os.path.join(plot_dir, f"{dataset_name}_full_benchmark.svg")

        plt.savefig(base_path_png, dpi=300)
        plt.savefig(base_path_svg)
        
        plt.show()
        plt.close()

        print(f"✅ Plot saved to {base_path_png}")

# ---------------------------------------------------------*/
# Plot results for a specific dataset folder
# ---------------------------------------------------------*/
def plot_results_for_dataset(dataset_result_dir):
    
    result_files = [f for f in os.listdir(dataset_result_dir) if f.endswith(".pkl")]
    if not result_files:
        print(f"⚠️ No results to plot in {dataset_result_dir}")
        return

    results = {}
    for result_file in result_files:
        optimizer_name = result_file.replace(".pkl", "")
        with open(os.path.join(dataset_result_dir, result_file), "rb") as f:
            results[optimizer_name] = pickle.load(f)

    optimizers = list(results.keys())

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    dataset_name = os.path.basename(dataset_result_dir)
    fig.suptitle(f"Benchmark Results - {dataset_name.upper()}", fontsize=20)

    # Final Test Accuracy
    ax = axs[0, 0]
    test_accuracies = [results[opt]["test_accuracy"] for opt in optimizers]
    sorted_data = sorted(zip(test_accuracies, optimizers))
    sorted_accuracies, sorted_optimizers = zip(*sorted_data)

    colors = [get_color(opt) for opt in sorted_optimizers]

    ax.bar(sorted_optimizers, sorted_accuracies, color=colors)
    ax.set_title("Final Test Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(sorted_optimizers)))
    ax.set_xticklabels(sorted_optimizers, rotation=45)
    ax.grid(axis="y")

    # Training Accuracy
    ax = axs[0, 1]
    for opt_name in optimizers:
        ax.plot(results[opt_name]["train_accuracies"], label=opt_name, color=get_color(opt_name))
    ax.set_title("Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid()
    ax.set_xticks(range(len(results[optimizers[0]]["train_accuracies"])))  # Nur ganzzahlige Labels
    
    # Training Loss
    ax = axs[1, 0]
    for opt_name in optimizers:
        ax.plot(results[opt_name]["train_losses"], label=opt_name, color=get_color(opt_name))
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    ax.set_xticks(range(len(results[optimizers[0]]["train_losses"])))  # Nur ganzzahlige Labels
    
    # Learning Rate
    ax = axs[1, 1]
    for opt_name in optimizers:
        ax.plot(results[opt_name]["lr_history"], label=opt_name, color=get_color(opt_name))
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.grid()
    ax.set_xticks(range(len(results[optimizers[0]]["lr_history"])))  # Nur ganzzahlige Labels

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path_png = os.path.join(dataset_result_dir, "benchmark_plot.png")
    save_path_svg = os.path.join(dataset_result_dir, "benchmark_plot.svg")

    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_svg)
    plt.show()
    plt.close()

    print(f"✅ Saved benchmark plot to {save_path_png}")


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\