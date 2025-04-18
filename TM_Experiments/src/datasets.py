# ---------------------------------------------------------*\
# Title: Dataset Loader
# Author: TM
# ---------------------------------------------------------*/

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn import datasets as sklearn_datasets
import numpy as np
from datasets import load_dataset as hf_load_dataset  # Huggingface datasets
from transformers import AutoTokenizer


# ---------------------------------------------------------*/
# Custom Dataset for tabular (LIBSVM) datasets
# ---------------------------------------------------------*/
class TabularDataset(Dataset):
    """Simple wrapper for tabular datasets (e.g., LIBSVM datasets)."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Assuming classification

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------*/
# Dataset Loader
# ---------------------------------------------------------*/
def load_dataset(dataset_name, data_dir="./data", batch_size=128, subset_percent=100):
    """Loads and prepares datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, LIBSVM datasets.

    Args:
        dataset_name (str): Name of the dataset ("mnist", "fmnist", "cifar10", "cifar100", "breast_cancer", etc.).
        data_dir (str): Directory to store datasets (for vision datasets).
        batch_size (int): Batch size for DataLoader.
        subset_percent (int): Percentage of dataset to use.

    Returns:
        train_loader, test_loader (DataLoader): Pytorch DataLoaders for vision or tabular datasets.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fmnist", "cifar10", "cifar100"]:
        # Vision datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if dataset_name == "mnist":
            train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

        elif dataset_name == "fmnist":
            train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
            test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

        elif dataset_name == "cifar10":
            train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

        elif dataset_name == "cifar100":
            train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
            test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name in ["breast_cancer", "iris", "wine", "digits"]:
        # Tabular datasets from sklearn
        if dataset_name == "breast_cancer":
            data = sklearn_datasets.load_breast_cancer()
        elif dataset_name == "iris":
            data = sklearn_datasets.load_iris()
        elif dataset_name == "wine":
            data = sklearn_datasets.load_wine()
        elif dataset_name == "digits":
            data = sklearn_datasets.load_digits()
        else:
            raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

        X = data.data
        y = data.target

        # Train/Test Split
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        split = int(0.8 * num_samples)

        train_indices = indices[:split]
        test_indices = indices[split:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        train_set = TabularDataset(X_train, y_train)
        test_set = TabularDataset(X_test, y_test)

    elif dataset_name in ["wikitext", "bookcorpus"]:
        # Load Huggingface language datasets
        if dataset_name == "wikitext":
            dataset = hf_load_dataset("wikitext", "wikitext-2-raw-v1")
        elif dataset_name == "bookcorpus":
            dataset = hf_load_dataset("bookcorpus")

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')  # oder dein Modell

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Important for padding!

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Define a simple PyTorch Dataset
        class HuggingfaceTextDataset(Dataset):
            def __init__(self, hf_dataset):
                self.dataset = hf_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val) for key, val in self.dataset[idx].items()}
                return item

        train_set = HuggingfaceTextDataset(tokenized_datasets["train"])
        test_set = HuggingfaceTextDataset(tokenized_datasets["test"])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        print(f"✅ Tokenized and loaded language dataset: {dataset_name.upper()}")
        return train_loader, test_loader


    
    #---------------------------------------------------------*/
    # Subset Option
    #---------------------------------------------------------*/
    if subset_percent < 100:
        num_train = len(train_set)
        num_test = len(test_set)

        train_indices = np.random.choice(num_train, int(num_train * subset_percent / 100), replace=False)
        test_indices = np.random.choice(num_test, int(num_test * subset_percent / 100), replace=False)

        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

        print(f"⚡ Using {subset_percent}% of the training and test datasets.")

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# -------------------------Notes-----------------------------------------------*\
# Supports: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 (vision) + LIBSVM datasets (breast_cancer, iris, wine, digits).
# LIBSVM datasets are tabular -> simple Logistic Regression benchmarking possible.
# -----------------------------------------------------------------------------*/
