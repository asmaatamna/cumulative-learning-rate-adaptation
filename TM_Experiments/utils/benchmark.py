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
import wandb
import logging
import multiprocessing
import os
import os
import pickle
import matplotlib.pyplot as plt
import wandb
import numpy as np

# ---------------------------------------------------------*/
# Wand Handling
# ---------------------------------------------------------*/
os.environ["WANDB_SILENT"] = "false"

# ---------------------------------------------------------*/
# Logger
# ---------------------------------------------------------*/
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/
OPTIMIZER_NAMES = [
    "SGD",
    "SGDMomentum",
    "Adam",
    "AdamW",
    "RMSProp",
    "Adam_CLARA",
    "Adam_CLARA_us",
    "SGD_CLARA",
    "SGD_CLARA_us"
]

# Fallback if a color is missing
DEFAULT_COLOR = "gray"

# Use a colormap (e.g., 'tab10', 'viridis', 'plasma')
colormap = plt.get_cmap('tab10', len(OPTIMIZER_NAMES))
OPTIMIZER_COLORS = {}
for i, name in enumerate(OPTIMIZER_NAMES):
    rgba = colormap(i)
    OPTIMIZER_COLORS[name] = tuple(rgba[:4])  # Konvertiert zu Python-Tuple (r, g, b, a)


def get_color(optimizer_name):
    return OPTIMIZER_COLORS.get(optimizer_name, DEFAULT_COLOR)

# ---------------------------------------------------------*/
# Benchmarking
# ---------------------------------------------------------*/


def run_optimizer_benchmark(dataset_name, optimizers, batch_size, num_classes, epochs, learning_rate, damping, seed, 
                            save_dir, subset=100, experiment_name=None):

    logger = logging.getLogger()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

        logger.info(f"Starte Optimizer {optimizer_name} — LR={learning_rate}, Seed={seed}")

        # 1) initialize W&B for exactly this (seed, damping, lr, optimizer) combo
        wandb.init(
            entity="cumulative-learning-rate-adaptation",  # <--- HIER festlegen
            project=experiment_name,
            group="optimizer-benchmark",              # optional: groups them under one experiment
            name=f"{dataset_name}_{optimizer_name}_lr{learning_rate:.0e}_d{damping}_s{seed}",
            config={
                "dataset": dataset_name,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "damping": damping,
                "seed": seed,
                "batch_size": batch_size,
                "epochs": epochs,
                "subset": subset,
            }
        )
        config = wandb.config

        print(f"\n----- Training with optimizer: {optimizer_name} -----")
        print(f"\nRunning benchmark on {device.upper()} with dataset: {dataset_name.upper()}\n")

        if is_language_model:
            model = load_model(dataset_name, num_classes=num_classes, input_dim=input_dim, model_type="tinytransformer")
        else:
            model = load_model(dataset_name, num_classes=num_classes, input_dim=input_dim)

        model.to(device)

        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate=learning_rate, damping=damping)

        # ⚡ Pass is_language_model here!
        init_loss, init_accuracy = evaluate_model(
            model,
            test_loader,
            device=device,
            is_language_model=is_language_model  # <-- HIER
        )

        wandb.log({
            "epoch": 0,
            "train_loss": init_loss,
            "train_accuracy": init_accuracy,
            "learning_rate": learning_rate
        })

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

        # Log training metrics to W&B
        for epoch, (loss, accuracy, lr) in enumerate(zip(t_losses, t_accuracies, t_lr)):
            wandb.log({
                "epoch": epoch + 1,
                "optimizer": optimizer_name,
                "dataset": dataset_name,
                "learning_rate": lr,
                "train_loss": loss,
                "train_accuracy": accuracy,
            })

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

        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        # Log final evaluation metrics to W&B
        wandb.log({
            "final_test_loss": test_loss,
            "final_test_accuracy": test_accuracy,
        })

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

        print(f"✅ Saved benchmark for {optimizer_name.upper()}")

        # Finish W&B run for this optimizer
        wandb.finish()

    elapsed_time = time.time() - start_time
    print(
        f"⏱️ Benchmark for {dataset_name.upper()} completed in {int(elapsed_time // 60)} min {elapsed_time % 60:.2f} sec")


def run_for_lr(lr, damping, result_dir, dataset, batch_size, num_classes, optimizers, epochs, seed, subset, experiment_name):

    for optimizer in optimizers:
        run_optimizer_benchmark(
            dataset_name=dataset,
            optimizers=[optimizer],
            batch_size=batch_size,
            num_classes=num_classes,
            epochs=epochs,
            learning_rate=lr,
            damping=damping,
            seed=seed,
            save_dir=result_dir,
            subset=subset,
            experiment_name=experiment_name,
        )
        # ⚡ After all optimizers finished -> Plot all results for this dataset
        # plot_results_for_dataset(result_dir, dataset_name=dataset)


# ---------------------------------------------------------*/
# Plot results for a specific dataset folder
# ---------------------------------------------------------*/


def to_tuple(c):
    if isinstance(c, np.ndarray):
        return tuple(c.tolist())
    elif isinstance(c, (list, tuple)):
        return tuple(c)
    return c  # z.B. String wie "gray"


def plot_results_for_dataset(dataset_result_dir, dataset_name=None):

    # Load result files
    result_files = [f for f in os.listdir(dataset_result_dir) if f.endswith(".pkl")]
    if not result_files:
        print(f"⚠️ No results to plot in {dataset_result_dir}")
        return

    # Deserialize into dict
    results = {}
    for result_file in result_files:
        opt = result_file.replace(".pkl", "")
        with open(os.path.join(dataset_result_dir, result_file), "rb") as f:
            results[opt] = pickle.load(f)
    optimizers = list(results.keys())

    if dataset_name is None:
        dataset_name = os.path.basename(dataset_result_dir)

    # Build figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Benchmark Results - {dataset_name.upper()}", fontsize=20)

    # Helper to cast colors
    def cast_color(c):
        if hasattr(c, "__iter__") and not isinstance(c, tuple):
            return tuple(float(x) for x in c)
        return c

    # 1) Final Test Accuracy
    ax = axs[0, 0]
    tests = [(results[o]["test_accuracy"], o) for o in optimizers]
    tests.sort()
    accs, opts = zip(*tests)
    colors = [to_tuple(get_color(o)) for o in opts]
    ax.bar(opts, accs, color=colors)
    ax.set_title("Final Test Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(opts)))
    ax.set_xticklabels(opts, rotation=45)
    ax.grid(axis="y")

    # 2) Training Accuracy
    ax = axs[0, 1]
    for o in optimizers:
        ax.plot(results[o]["train_accuracies"], label=o, color=cast_color(get_color(o)))
    ax.set_title("Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid()
    ax.set_xticks(range(len(results[optimizers[0]]["train_accuracies"])))

    # 3) Training Loss
    ax = axs[1, 0]
    for o in optimizers:
        ax.plot(results[o]["train_losses"], label=o, color=cast_color(get_color(o)))
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    ax.set_xticks(range(len(results[optimizers[0]]["train_losses"])))

    # 4) Learning Rate
    ax = axs[1, 1]
    for o in optimizers:
        ax.plot(results[o]["lr_history"], label=o, color=cast_color(get_color(o)))
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    ax.set_xticks(range(len(results[optimizers[0]]["lr_history"])))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Log the figure to W&B
    wandb.log({
        f"{dataset_name}_benchmark_plot": wandb.Image(fig, caption=f"{dataset_name} results")
    })

    # Save locally
    save_png = os.path.join(dataset_result_dir, "benchmark_plot.png")
    save_svg = os.path.join(dataset_result_dir, "benchmark_plot.svg")
    plt.savefig(save_png, dpi=300)
    plt.savefig(save_svg)
    plt.close(fig)

    print(f"✅ Saved and logged benchmark plot to W&B and {save_png}")


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
