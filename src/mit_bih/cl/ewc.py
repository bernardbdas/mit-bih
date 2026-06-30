"""
Elastic Weight Consolidation (EWC) utilities for continual learning.

This module provides functions to calculate parameter importances (Fisher Information Matrices)
and EWC regularization loss penalties to mitigate catastrophic forgetting during sequential training tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_fisher(
    model: nn.Module,
    dataloader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device | str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Computes the diagonal Fisher Information Matrix and parameter optimal states
    for all trainable parameters in the model.

    Args:
        model: The model to compute parameter importance for.
        dataloader: DataLoader containing the training data of the current task.
        criterion: The loss function used during training.
        device: The computing device (cpu or cuda).

    Returns:
        A tuple of (fisher, opt_par):
            - fisher: Dict mapping parameter names to their calculated importance tensors.
            - opt_par: Dict mapping parameter names to their optimal values at the end of the task.
    """
    if dataloader is None:
        return {}, {}

    fisher: dict[str, torch.Tensor] = {}
    opt_par: dict[str, torch.Tensor] = {}

    # Initialize Fisher matrices and save optimal parameter parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            opt_par[name] = param.data.clone()

    model.eval()
    n_batches = len(dataloader)
    if n_batches == 0:
        return fisher, opt_par

    # Backpropagate gradients to estimate empirical Fisher values
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        
        # Calculate loss and backward gradients
        loss = criterion(model(inputs), labels)
        loss.backward()
        
        # Accumulate squared gradients as a proxy for parameter sensitivity
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2) / n_batches

    return fisher, opt_par


def ewc_penalty(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
    opt_par: dict[str, torch.Tensor],
    ewc_lambda_full: float,
    ewc_lambda_lora: float,
    device: torch.device | str
) -> torch.Tensor:
    """
    Computes the Elastic Weight Consolidation (EWC) penalty term.

    The penalty restricts parameters from moving too far from their optimal values,
    weighted by their calculated Fisher importance.

    Args:
        model: The current model being trained.
        fisher: Pre-computed Fisher importance dictionary.
        opt_par: Pre-computed optimal parameter dictionary from previous tasks.
        ewc_lambda_full: EWC weight multiplier for full-parameter training.
        ewc_lambda_lora: EWC weight multiplier for LoRA-based parameter training.
        device: The computing device.

    Returns:
        A scalar tensor representing the EWC regularization loss.
    """
    if not fisher:
        return torch.tensor(0.0, device=device)

    # Determine regularization weight based on model parameter configuration
    ewc_lambda = ewc_lambda_lora if getattr(model, "use_lora", False) else ewc_lambda_full

    # Normalize Fisher values so that the scale of the penalty remains bounded
    max_fisher = max(
        (f.max().item() for f in fisher.values() if f.numel() > 0),
        default=0.0
    )
    norm = max_fisher if max_fisher > 1e-12 else 1.0

    loss = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher:
            # Quadratic distance from optimal parameters scaled by normalized Fisher
            loss += (
                (fisher[name].to(device) / norm) *
                (param - opt_par[name].to(device)).pow(2)
            ).sum()
            
    return (ewc_lambda / 2.0) * loss
