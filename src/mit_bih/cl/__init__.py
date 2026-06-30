"""
Continual learning (CL) algorithms and helper functions.
"""

from mit_bih.cl.ewc import compute_fisher, ewc_penalty
from mit_bih.cl.trainer import (
    FocalLoss,
    EarlyStopping,
    train_local_continual,
    evaluate_model
)

__all__ = [
    "compute_fisher",
    "ewc_penalty",
    "FocalLoss",
    "EarlyStopping",
    "train_local_continual",
    "evaluate_model"
]
