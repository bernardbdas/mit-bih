"""
Unit tests for Continual Learning (CL) and Federated Learning (FL) algorithms.
"""

import copy
import torch
import torch.nn as nn
from mit_bih.models import ECGCNN
from mit_bih.cl import FocalLoss, compute_fisher, ewc_penalty, train_local_continual
from mit_bih.fl import (
    federated_averaging,
    federated_average_dict,
    federated_median,
    federated_trimmed_mean
)


def test_focal_loss():
    """Verify FocalLoss computes valid loss values."""
    criterion = FocalLoss(gamma=2.0)
    # Correctly aligned predictions (logits) vs target
    inputs = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss_easy = criterion(inputs, targets)
    
    # Misaligned predictions
    inputs_bad = torch.tensor([[-10.0, 10.0], [10.0, -10.0]], dtype=torch.float32)
    loss_hard = criterion(inputs_bad, targets)
    
    # Misaligned predictions should yield much higher loss
    assert loss_easy.item() < loss_hard.item()
    assert loss_easy.item() >= 0.0


def test_ewc_and_fisher_computation(synthetic_dataloader):
    """Verify Fisher Information matrix computation and EWC penalty calculation."""
    model = ECGCNN(use_lora=False)
    criterion = FocalLoss(gamma=2.0)
    device = "cpu"
    
    fisher, opt_par = compute_fisher(model, synthetic_dataloader, criterion, device)
    
    # Verify we get importance estimates for trainable parameters
    assert len(fisher) > 0
    assert len(opt_par) > 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in fisher
            assert name in opt_par
            assert fisher[name].shape == param.shape
            assert opt_par[name].shape == param.shape

    # Compute EWC penalty
    penalty = ewc_penalty(
        model=model,
        fisher=fisher,
        opt_par=opt_par,
        ewc_lambda_full=100.0,
        ewc_lambda_lora=1.0,
        device=device
    )
    # Penalty should be a positive scalar tensor
    assert isinstance(penalty, torch.Tensor)
    assert penalty.ndim == 0
    # Because parameters have not changed from opt_par, penalty should be close to 0
    assert torch.allclose(penalty, torch.tensor(0.0), atol=1e-5)


def test_federated_averaging():
    """Verify FedAvg state dict aggregation behaves correctly."""
    model = ECGCNN(use_lora=False)
    w_init = copy.deepcopy(model.state_dict())
    
    w1 = copy.deepcopy(w_init)
    w2 = copy.deepcopy(w_init)
    
    # Modify weights
    for k in w1.keys():
        if w1[k].is_floating_point():
            w1[k] = w1[k] + 1.0
            w2[k] = w2[k] + 2.0
            
    # Aggregate with equal client sizes (10 and 10)
    agg = federated_averaging(model, [w1, w2], [10, 10])
    
    # Averaged weight should be offset by 1.5
    for k in agg.keys():
        if agg[k].is_floating_point():
            expected = w_init[k] + 1.5
            assert torch.allclose(agg[k], expected, atol=1e-5)


def test_federated_average_dict():
    """Verify dictionary aggregation ignoring zero-sample clients."""
    d1 = {'w': torch.tensor([1.0, 2.0])}
    d2 = {'w': torch.tensor([3.0, 4.0])}
    d3 = {'w': torch.tensor([10.0, 20.0])}
    
    # client 3 has size 0 (skipped), clients 1 & 2 have sizes 1 and 3
    agg = federated_average_dict([d1, d2, d3], [1, 3, 0])
    
    # Expected weighted avg: d1 * (1/4) + d2 * (3/4) = 0.25 * [1, 2] + 0.75 * [3, 4] = [0.25+2.25, 0.5+3] = [2.5, 3.5]
    assert torch.allclose(agg['w'], torch.tensor([2.5, 3.5]))


def test_train_local_continual(synthetic_dataloader):
    """Verify that local training updates model parameters."""
    model = ECGCNN(use_lora=False)
    w_before = copy.deepcopy(model.state_dict())
    criterion = FocalLoss(gamma=2.0)
    
    # Train for 1 epoch
    trained_model = train_local_continual(
        model=model,
        train_loader=synthetic_dataloader,
        val_loader=synthetic_dataloader,
        criterion=criterion,
        local_epochs=1,
        lr=0.1,
        patience=1,
        use_ewc=False,
        device="cpu"
    )
    
    w_after = trained_model.state_dict()
    
    # Verify parameters changed
    diff = False
    for k in w_before.keys():
        if not torch.allclose(w_before[k], w_after[k]):
            diff = True
            break
    assert diff, "Parameters did not change after training!"


def test_federated_median():
    """Verify FedMedian correctly selects the median element and preserves dtypes."""
    # 3 clients
    d1 = {'w': torch.tensor([1.0, 10.0]), 'b': torch.tensor([2, 5], dtype=torch.long)}
    d2 = {'w': torch.tensor([2.0, 1.0]), 'b': torch.tensor([0, 8], dtype=torch.long)}
    d3 = {'w': torch.tensor([5.0, 2.0]), 'b': torch.tensor([1, 6], dtype=torch.long)}
    
    agg = federated_median([d1, d2, d3])
    
    # Median of [1, 2, 5] is 2. Median of [10, 1, 2] is 2.
    assert torch.allclose(agg['w'], torch.tensor([2.0, 2.0]))
    
    # Median of [2, 0, 1] is 1. Median of [5, 8, 6] is 6.
    # Type should be preserved as torch.long
    assert torch.equal(agg['b'], torch.tensor([1, 6], dtype=torch.long))


def test_federated_trimmed_mean():
    """Verify FedTrimmedMean trims outlier values and computes correct average."""
    # 5 clients. Beta = 0.2 means k = int(5 * 0.2) = 1 client trimmed from each end
    # Sliced parameters sorted: sorted[1:4] (indices 1, 2, 3)
    d1 = {'w': torch.tensor([1.0, 100.0])}  # 100.0 is an outlier
    d2 = {'w': torch.tensor([2.0, 2.0])}
    d3 = {'w': torch.tensor([3.0, 3.0])}
    d4 = {'w': torch.tensor([4.0, 4.0])}
    d5 = {'w': torch.tensor([-50.0, 5.0])}  # -50.0 is an outlier
    
    # Sorted for coord 0: [-50.0, 1.0, 2.0, 3.0, 4.0]. Trimmed: [1.0, 2.0, 3.0]. Mean = (1+2+3)/3 = 2.0
    # Sorted for coord 1: [2.0, 3.0, 4.0, 5.0, 100.0]. Trimmed: [3.0, 4.0, 5.0]. Mean = (3+4+5)/3 = 4.0
    agg = federated_trimmed_mean([d1, d2, d3, d4, d5], beta=0.2)
    
    assert torch.allclose(agg['w'], torch.tensor([2.0, 4.0]))

