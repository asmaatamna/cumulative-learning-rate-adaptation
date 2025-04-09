# ---------------------------------------------------------*\
# Title: Training Loop
# Author: TM
# ---------------------------------------------------------*/

import torch
import torch.nn as nn

def train_model(model, train_loader, optimizer, epochs=10, device="cpu"):
    """Trains a model on the given data.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        epochs (int): Number of training epochs.
        device (str): Device to train on ("cpu" or "cuda").

    Returns:
        train_losses (list): List of training loss per epoch.
        train_accuracies (list): List of training accuracy per epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return train_losses, train_accuracies

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\