# ---------------------------------------------------------*\
# Title: Model Evaluation
# Author: TM
# ---------------------------------------------------------*/

import torch
import torch.nn as nn

def evaluate_model(model, test_loader, device="cpu"):
    """Evaluates a trained model on the test data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): Test data loader.
        device (str): Device to evaluate on ("cpu" or "cuda").

    Returns:
        test_loss (float): Average loss on the test set.
        test_accuracy (float): Accuracy on the test set.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100. * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return test_loss, test_accuracy


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\