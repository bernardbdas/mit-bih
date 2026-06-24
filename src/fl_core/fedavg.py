import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def federated_averaging(global_model, client_weights, client_sizes):
    """
    Computes FedAvg: weighted average of client weights.
    """
    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()
    
    # Clone global dict structure and set weights to zero
    aggregated_dict = copy.deepcopy(global_dict)
    for key in aggregated_dict.keys():
        aggregated_dict[key] = torch.zeros_like(aggregated_dict[key], dtype=torch.float)
        
    # Compute weighted averages
    for client_dict, size in zip(client_weights, client_sizes):
        weight = size / total_samples
        for key in aggregated_dict.keys():
            aggregated_dict[key] += client_dict[key].to(aggregated_dict[key].device) * weight
            
    return aggregated_dict

def train_local_client(model, dataloader, epochs=1, lr=0.001, device="cpu"):
    """
    Runs local training updates on client data.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluates the model on test data, returning accuracy, actual labels, and predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds), all_labels, all_preds

def federated_average_dict(local_weights, client_sizes):
    """
    Weighted FedAvg aggregation of state dictionaries or parameter dicts.
    Clients with zero data for the current task (size 0) are excluded from the weighting.
    """
    weights_and_sizes = [
        (w, s) for w, s in zip(local_weights, client_sizes) if s > 0
    ]
    if not weights_and_sizes:
        # no client had data this round; nothing to aggregate
        return local_weights[0]

    total = sum(s for _, s in weights_and_sizes)
    agg = {}
    for key in weights_and_sizes[0][0].keys():
        agg[key] = sum(
            (s / total) * w[key].float()
            for w, s in weights_and_sizes
        )
    return agg

