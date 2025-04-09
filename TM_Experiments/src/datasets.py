# ---------------------------------------------------------*\
# Title: Dataset Loader
# Author: TM
# ---------------------------------------------------------*/

import os
import torch
from torchvision import datasets, transforms
from datasets import load_dataset  # Huggingface datasets


def load_dataset(dataset_name, data_dir="./data", batch_size=128):
    """Loads and prepares datasets: MNIST, Fashion-MNIST, CIFAR-10, TinyStories.

    Args:
        dataset_name (str): Name of the dataset ("mnist", "fmnist", "cifar10", "tinystories").
        data_dir (str): Directory to store datasets.
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader, test_loader (DataLoader): Pytorch DataLoaders for vision datasets.
        OR
        train_texts, test_texts (List[str]): Text data for TinyStories.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fmnist", "cifar10"]:
        # Transformation for vision datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if dataset_name == "mnist":
        dataset_path = os.path.join(data_dir, "MNIST")
        if os.path.exists(dataset_path):
            print("MNIST already downloaded.")
        else:
            print("Downloading MNIST...")
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "fmnist":
        dataset_path = os.path.join(data_dir, "FashionMNIST")
        if os.path.exists(dataset_path):
            print("Fashion-MNIST already downloaded.")
        else:
            print("Downloading Fashion-MNIST...")
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "cifar10":
        dataset_path = os.path.join(data_dir, "CIFAR10")
        if os.path.exists(dataset_path):
            print("CIFAR-10 already downloaded.")
        else:
            print("Downloading CIFAR-10...")
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # For vision datasets, return DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# -------------------------Notes-----------------------------------------------*\
# Handles Vision datasets via torchvision.
# Handles TinyStories via Huggingface Datasets.
# TinyStories uses 'train' and 'validation' splits (no 'test' split).
# -----------------------------------------------------------------------------*\
