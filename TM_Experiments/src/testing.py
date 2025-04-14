# ---------------------------------------------------------*\
# Title: Model Evaluation
# Author: TM
# ---------------------------------------------------------*/

import torch
import torch.nn as nn

def evaluate_model(model, test_loader, device="cpu", is_language_model=False):
    """Evaluates a trained model on the test data."""

    model.to(device)
    model.eval()
    
    if is_language_model:
        print("⚡ Skipping evaluation: Language model detected (no classification).")
        return 0.0, 0.0  # Dummy values
    else:
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
