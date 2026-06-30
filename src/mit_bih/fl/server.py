"""
Federated learning server aggregation algorithms.

This module provides routines to perform federated averaging (FedAvg) over
parameter dictionaries, supporting unequal client dataset sizes and client exclusion
when partitions contain zero data points for a given task.
"""

import copy
import torch
import torch.nn as nn


def federated_averaging(
    global_model: nn.Module,
    client_weights: list[dict[str, torch.Tensor]],
    client_sizes: list[int]
) -> dict[str, torch.Tensor]:
    """
    Computes standard FedAvg: a weighted average of client state dictionaries.

    Ensures that integer buffers (e.g., batch norm tracking stats) are correctly aggregated
    and cast back to their original dtype (with rounding) to prevent type pollution.

    Args:
        global_model: The global reference model structure.
        client_weights: List of state dictionaries from trained clients.
        client_sizes: Number of samples processed by each client during local training.

    Returns:
        The consolidated global state dictionary.
    """
    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()
    
    # Initialize accumulator dictionary in float for precision
    acc_dict = {}
    for key in global_dict.keys():
        acc_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float)
        
    # Perform weighted accumulation of client weights in float
    for client_dict, size in zip(client_weights, client_sizes):
        weight = size / total_samples
        for key in acc_dict.keys():
            acc_dict[key] += client_dict[key].to(acc_dict[key].device).float() * weight
            
    # Copy back to global_dict, converting non-float variables (e.g., tracking stats) back to original dtypes
    for key in global_dict.keys():
        if not global_dict[key].is_floating_point():
            global_dict[key].copy_(acc_dict[key].round().to(global_dict[key].dtype))
        else:
            global_dict[key].copy_(acc_dict[key])
            
    return global_dict


def federated_average_dict(
    local_weights: list[dict[str, torch.Tensor]],
    client_sizes: list[int]
) -> dict[str, torch.Tensor]:
    """
    Weighted FedAvg aggregation of client parameter dictionaries.

    Safely excludes clients that did not process any training data (dataset size 0)
    for the current task/round to prevent divide-by-zero errors.

    Args:
        local_weights: List of parameter dictionaries (could be full model state dicts
                       or LoRA adapter weights) from participating clients.
        client_sizes: List of sample sizes processed by each client this round.

    Returns:
        The averaged parameter dictionary.
    """
    # Filter out client dictionaries where sample counts are zero
    weights_and_sizes = [
        (w, s) for w, s in zip(local_weights, client_sizes) if s > 0
    ]
    
    if not weights_and_sizes:
        # If no clients participated, return the first dictionary as fallback
        return local_weights[0]

    total = sum(s for _, s in weights_and_sizes)
    agg = {}
    
    # Average across participating clients
    for key in weights_and_sizes[0][0].keys():
        agg[key] = sum(
            (s / total) * w[key].float()
            for w, s in weights_and_sizes
        )
        
    return agg


def federated_median(
    local_weights: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """
    Computes coordinate-wise FedMedian: the median of client parameters.

    Robust to outliers and malicious updates.

    Args:
        local_weights: List of parameter dictionaries from participating clients.

    Returns:
        The aggregated parameter dictionary.
    """
    if not local_weights:
        raise ValueError("Cannot aggregate empty list of weights.")

    agg = {}
    # Iterate over parameter names
    for key in local_weights[0].keys():
        # Stack parameters along client dimension (dim=0) in float
        stacked_tensors = torch.stack([w[key].float() for w in local_weights], dim=0)
        
        # Calculate coordinate-wise median
        median_tensor = torch.median(stacked_tensors, dim=0).values
        
        # Restore original integer types if applicable (e.g. tracking buffers)
        original_tensor = local_weights[0][key]
        if not original_tensor.is_floating_point():
            agg[key] = median_tensor.round().to(original_tensor.dtype)
        else:
            agg[key] = median_tensor
            
    return agg


def federated_trimmed_mean(
    local_weights: list[dict[str, torch.Tensor]],
    beta: float = 0.2
) -> dict[str, torch.Tensor]:
    """
    Computes coordinate-wise FedTrimmedMean.

    Trims a fraction beta of the largest and smallest values for each parameter
    coordinate across clients, and averages the remaining ones.

    Args:
        local_weights: List of parameter dictionaries from participating clients.
        beta: The fraction of values to trim from each end (0.0 <= beta < 0.5).

    Returns:
        The aggregated parameter dictionary.
    """
    if not local_weights:
        raise ValueError("Cannot aggregate empty list of weights.")
    if not (0.0 <= beta < 0.5):
        raise ValueError("Trimming fraction beta must satisfy 0.0 <= beta < 0.5")

    num_clients = len(local_weights)
    k = int(num_clients * beta)
    
    # If trimming is too aggressive, trim at most (N - 1) // 2 from each end
    if 2 * k >= num_clients:
        k = max(0, (num_clients - 1) // 2)

    agg = {}
    for key in local_weights[0].keys():
        # Stack parameters along client dimension (dim=0) in float
        stacked_tensors = torch.stack([w[key].float() for w in local_weights], dim=0)
        
        # Sort along client dimension
        sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)
        
        # Slice and average the remaining non-trimmed values
        trimmed_mean_tensor = torch.mean(sorted_tensors[k : num_clients - k], dim=0)
        
        # Restore original integer types if applicable
        original_tensor = local_weights[0][key]
        if not original_tensor.is_floating_point():
            agg[key] = trimmed_mean_tensor.round().to(original_tensor.dtype)
        else:
            agg[key] = trimmed_mean_tensor
            
    return agg
