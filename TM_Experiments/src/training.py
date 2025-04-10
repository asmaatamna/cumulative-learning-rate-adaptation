# ---------------------------------------------------------*\
# Title: Training Loop
# Author: TM
# ---------------------------------------------------------*/

import torch.nn as nn

def train_model(model, train_loader, optimizer, epochs=10, device="cpu"):
    """Trains a model and stores training loss, accuracy and current learning rate per epoch."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    lr_history = []  # Hier speichern wir die aktuelle Lernrate pro Epoche

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

            # Berechne Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Erfasse die aktuelle Lernrate (angenommen, alle Parametergruppen haben denselben lr)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, LR: {current_lr:.6f}")

    return train_losses, train_accuracies, lr_history


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\