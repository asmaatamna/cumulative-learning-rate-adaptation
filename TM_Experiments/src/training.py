# ---------------------------------------------------------*\
# Title: Training Loop
# Author: TM
# ---------------------------------------------------------*/

import torch
import torch.nn as nn

def train_model(model, train_loader, optimizer, epochs=10, device="cpu"):
    """Trains a model and stores training loss, accuracy and current learning rate per epoch."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    lr_history = []

    is_language_model = hasattr(model, "transformer")  # ⚡ quick check if it's a Huggingface model

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if is_language_model:
                # Huggingface models (e.g., DistilGPT2)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                # Huggingface models automatically compute loss if labels=input_ids
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs.loss
            else:
                # Vision / Tabular datasets
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # Backward and optimizer step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        if not is_language_model:
            epoch_accuracy = 100. * correct / total
            train_accuracies.append(epoch_accuracy)
        else:
            train_accuracies.append(0)  # Placeholder (language modeling accuracy is tricky)

        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.2f}, "
              f"{'Accuracy: ' + str(f'{epoch_accuracy:.2f}') + '%' if not is_language_model else 'Language Model'} "
              f"LR: {current_lr:.2e}")

    return train_losses, train_accuracies, lr_history


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\