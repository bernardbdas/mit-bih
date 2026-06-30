"""
Pytest configuration and shared test fixtures.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def synthetic_ecg_data():
    """Generates synthetic ECG data for testing."""
    np.random.seed(42)
    # Generate 100 samples of length 180
    X = np.random.randn(100, 180).astype(np.float32)
    # Generate random class labels (0 to 4)
    y = np.random.randint(0, 5, size=(100,)).astype(np.int64)
    return X, y


@pytest.fixture
def synthetic_dataloader(synthetic_ecg_data):
    """Generates a DataLoader with synthetic ECG data."""
    X, y = synthetic_ecg_data
    Xt = torch.tensor(X).unsqueeze(1)  # (100, 1, 180)
    yt = torch.tensor(y)
    dataset = TensorDataset(Xt, yt)
    return DataLoader(dataset, batch_size=10, shuffle=False)
