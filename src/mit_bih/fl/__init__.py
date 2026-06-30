"""
Federated learning (FL) algorithms, client wrappers, and simulation runners.
"""

from mit_bih.fl.client import (
    train_local_client,
    get_weights,
    set_weights,
    ArrhythmiaFlowerClient
)
from mit_bih.fl.server import (
    federated_averaging,
    federated_average_dict,
    federated_median,
    federated_trimmed_mean
)
from mit_bih.fl.simulation import (
    pretrain_backbone,
    run_local_simulation,
    run_flower_simulation
)

__all__ = [
    "train_local_client",
    "get_weights",
    "set_weights",
    "ArrhythmiaFlowerClient",
    "federated_averaging",
    "federated_average_dict",
    "federated_median",
    "federated_trimmed_mean",
    "pretrain_backbone",
    "run_local_simulation",
    "run_flower_simulation"
]
