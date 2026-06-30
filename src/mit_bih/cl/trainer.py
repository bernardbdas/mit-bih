"""
Continual learning training loops and performance evaluation tools.

This module provides training procedures for local clients incorporating
Elastic Weight Consolidation (EWC) penalties, early stopping hooks,
and comprehensive model evaluation using standard classification metrics.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from mit_bih.cl.ewc import ewc_penalty


class FocalLoss(nn.Module):
    """
    Focal Loss function to address heavy class imbalance.

    Focal Loss dynamically down-weights well-classified/easy examples
    and focuses gradient updates on rare or hard examples.
    
    Formula:
        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma: float = 2.0):
        """
        Initializes FocalLoss.

        Args:
            gamma: Focusing parameter to scale hard examples vs easy ones.
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the focal loss.

        Args:
            inputs: Model logits of shape (batch_size, num_classes).
            targets: Target labels of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class EarlyStopping:
    """
    Early stopping helper to prevent overfitting during local client training.
    """
    def __init__(self, patience: int = 5):
        """
        Initializes EarlyStopping.

        Args:
            patience: Number of validation check epochs to wait for improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False

    def __call__(self, val_loss: float) -> None:
        """
        Checks validation loss and updates early stopping state.

        Args:
            val_loss: Calculated validation loss for the current epoch.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_local_continual(
    model: nn.Module,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    criterion: nn.Module,
    local_epochs: int = 10,
    lr: float = 0.001,
    patience: int = 5,
    use_ewc: bool = False,
    fisher: dict[str, torch.Tensor] | None = None,
    opt_par: dict[str, torch.Tensor] | None = None,
    ewc_lambda_full: float = 50.0,
    ewc_lambda_lora: float = 1.0,
    device: torch.device | str = "cpu"
) -> nn.Module:
    """
    Trains a model locally on a specific client's partition for the current task.

    Supports EWC penalties to regularize against forgetting previous tasks, early stopping,
    and adaptive learning rate scheduling.

    Args:
        model: The PyTorch neural network to train.
        train_loader: DataLoader containing the training partition.
        val_loader: DataLoader containing the validation partition.
        criterion: The loss function to minimize.
        local_epochs: Maximum epochs to train locally.
        lr: Initial learning rate.
        patience: Patience epochs for early stopping.
        use_ewc: If True, applies EWC penalty based on fisher and opt_par.
        fisher: Pre-computed Fisher Information matrix from previous tasks.
        opt_par: Pre-computed optimal parameter states from previous tasks.
        ewc_lambda_full: Regularization scale parameter for full training.
        ewc_lambda_lora: Regularization scale parameter for LoRA training.
        device: The computing device.

    Returns:
        The trained model with the lowest validation loss state.
    """
    if train_loader is None or len(train_loader) == 0:
        return model

    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    early_stopper = EarlyStopping(patience=patience)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(local_epochs):
        # Configure model modes (handling custom training hooks if present)
        if hasattr(model, "set_train_mode"):
            model.set_train_mode()
        else:
            model.train()

        running_loss = 0.0
        running_task_loss = 0.0
        running_ewc_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            task_loss = criterion(model(inputs), labels)
            loss = task_loss

            # Apply EWC regularization penalty
            ewc_loss_val = torch.tensor(0.0, device=device)
            if use_ewc and fisher:
                ewc_loss_val = ewc_penalty(
                    model, fisher, opt_par,
                    ewc_lambda_full, ewc_lambda_lora, device
                )
                loss = loss + ewc_loss_val

            loss.backward()
            
            # Clip gradients to ensure numerical stability
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0
            )
            optimizer.step()

            running_loss += loss.item()
            running_task_loss += task_loss.item()
            running_ewc_loss += ewc_loss_val.item()

        train_loss = running_loss / len(train_loader)
        avg_task_loss = running_task_loss / len(train_loader)
        avg_ewc_loss = running_ewc_loss / len(train_loader)

        # Run validation pass to monitor generalizability
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None and len(val_loader) > 0:
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for v_inputs, v_labels in val_loader:
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                    outputs = model(v_inputs)
                    total_loss += criterion(outputs, v_labels).item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == v_labels).sum().item()
                    total += v_labels.size(0)
            if total > 0:
                val_loss = total_loss / len(val_loader)
                val_acc = 100.0 * correct / total

        scheduler.step(val_loss)

        if use_ewc and avg_task_loss > 1e-8 and avg_ewc_loss > 5 * avg_task_loss:
            print(
                f"      WARNING: EWC penalty ({avg_ewc_loss:.4f}) >> task loss ({avg_task_loss:.4f}). "
                f"EWC may be dominating training."
            )

        print(
            f"      Epoch {epoch+1}/{local_epochs} | "
            f"Train Loss: {train_loss:.4f} (task={avg_task_loss:.4f} ewc={avg_ewc_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        # Verify early stopping condition
        early_stopper(val_loss)
        if early_stopper.stop:
            print("      Early stopping triggered.")
            break

    # Restore the model weights corresponding to the best validation score
    model.load_state_dict(best_state)
    return model


def evaluate_model(model: nn.Module, dataloader: DataLoader | None, device: torch.device | str = "cpu") -> dict | None:
    """
    Evaluates model performance across a dataset, calculating accuracy, precision,
    recall, F1 macro-averages, and confusion matrix.

    Args:
        model: Model to evaluate.
        dataloader: Dataset loader containing inputs and target labels.
        device: CPU or GPU execution target.

    Returns:
        A dictionary containing:
            - accuracy: float
            - precision: float (macro average)
            - recall: float (macro average)
            - f1: float (macro average)
            - cm: 5x5 numpy array representation of the confusion matrix
            - preds: List of predictions
            - true: List of actual labels
        Returns None if dataset is empty or None.
    """
    if dataloader is None or len(dataloader) == 0:
        return None

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    if len(y_true) == 0:
        return None

    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1"       : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cm"       : confusion_matrix(y_true, y_pred, labels=list(range(5))),
        "preds"    : y_pred,
        "true"     : y_true
    }
