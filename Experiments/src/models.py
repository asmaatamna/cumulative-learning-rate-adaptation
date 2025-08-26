# ---------------------------------------------------------*\
# Title: Model Loader
# Author: TM
# ---------------------------------------------------------*/

import torch.nn as nn
import timm  # For WideResNet and other models
from transformers import AutoModelForCausalLM  # For small language model (Huggingface)


def load_model(dataset_name, num_classes=10, input_dim=None, model_type="default"):
    """Loads a model depending on the dataset and task.

    Args:
        dataset_name (str): Name of the dataset.
        num_classes (int): Number of output classes.
        input_dim (int, optional): Input dimension for tabular datasets.
        model_type (str): Optional model type override ("default", "wideresnet", "tinytransformer").
    """

    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fmnist"]:
        # Small MLP for MNIST and Fashion-MNIST
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    elif dataset_name in ["cifar10", "cifar100"]:
        if model_type == "wideresnet":
            # WideResNet-16-8 from timm
            model = timm.create_model('wide_resnet16_8', pretrained=False, num_classes=num_classes)
        else:
            # Simple CNN
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
                nn.Linear(8 * 8 * 128, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

    elif dataset_name in ["breast_cancer", "iris", "wine", "digits"]:
        # Logistic Regression model for tabular data
        if input_dim is None:
            raise ValueError("Input dimension must be provided for tabular datasets.")
        
        model = nn.Sequential(
            nn.Linear(input_dim, num_classes)  # Logistic Regression (no activation)
        )

    elif dataset_name in ["tinystories", "wikitext", "language_toy"]:
        if model_type == "tinytransformer":
            # Load a small GPT-like model from Huggingface (e.g., distilgpt2)
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        else:
            raise ValueError("For language datasets, please specify model_type='tinytransformer'.")

    else:
        raise ValueError(f"No model defined for dataset {dataset_name}")

    return model

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\