#---------------------------------------------------------*\
# Title: 
# Author: 
#---------------------------------------------------------*/
import torch.nn as nn

def load_model(dataset_name, num_classes=10):
    """Loads a simple model depending on the dataset."""

    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fmnist"]:
        # Small MLP for MNIST and Fashion-MNIST
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif dataset_name == "cifar10":
        # Simple CNN for CIFAR-10
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8*8*128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"No model defined for dataset {dataset_name}")

    return model


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\