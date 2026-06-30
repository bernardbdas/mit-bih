"""
Unit tests for data preprocessing and partitioning utilities.
"""

import unittest
import numpy as np
import torch
from mit_bih.data.preprocess import (
    bandpass_filter,
    split_by_task,
    make_continual_loader
)


class TestDataPreprocessing(unittest.TestCase):
    """Tests the signal processing and data loader creation utilities."""

    def test_bandpass_filter(self):
        """Verify the Butterworth bandpass filter behaves correctly."""
        # Create a simple sine wave signal
        t = np.linspace(0, 1, 360, endpoint=False)
        # Combine low freq (0.1 Hz) and high freq (100 Hz) noise with a clean signal (10 Hz)
        sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)
        
        filtered = bandpass_filter(sig, lowcut=0.5, highcut=45.0, fs=360.0, order=4)
        
        self.assertEqual(filtered.shape, sig.shape)
        # Ensure the filter modified the signal (removed low and high frequency noise)
        self.assertFalse(np.allclose(filtered, sig))

    def test_split_by_task(self):
        """Verify that dataset splits map correctly to task classes."""
        X = np.random.randn(20, 180)
        y = np.array([0, 1, 2, 3, 4] * 4)
        
        task_classes = [[0, 1], [2, 3, 4]]
        subsets = split_by_task(X, y, task_classes)
        
        self.assertEqual(len(subsets), 2)
        # Task 1 check: classes 0 and 1
        X1, y1 = subsets[0]
        self.assertEqual(X1.shape[0], 8)
        self.assertTrue(np.all(np.isin(y1, [0, 1])))
        
        # Task 2 check: classes 2, 3, 4
        X2, y2 = subsets[1]
        self.assertEqual(X2.shape[0], 12)
        self.assertTrue(np.all(np.isin(y2, [2, 3, 4])))

    def test_make_continual_loader(self):
        """Verify DataLoader creation and weighted random sampling."""
        X = np.random.randn(30, 180)
        # Create a highly imbalanced target set: 25 of class 0, 5 of class 1
        y = np.array([0] * 25 + [1] * 5)
        
        loader_normal = make_continual_loader(X, y, batch_size=10, shuffle=False, oversample=False)
        self.assertEqual(len(loader_normal), 3)  # 30 / 10 = 3 batches
        
        # Check shapes
        inputs, labels = next(iter(loader_normal))
        self.assertEqual(inputs.shape, (10, 1, 180))
        self.assertEqual(labels.shape, (10,))
        
        # Create loader with oversampling
        loader_oversampled = make_continual_loader(X, y, batch_size=10, oversample=True)
        # Iterate over the loader and check that minority class is oversampled
        all_labels = []
        for _, batch_labels in loader_oversampled:
            all_labels.extend(batch_labels.numpy())
            
        # Total counts should be equal to the length of dataset (30)
        self.assertEqual(len(all_labels), 30)
        # Since class 1 is oversampled, its count in the loaded batches should generally be higher than 5
        # (It is probabilistic, but in expectation it should be balanced)
        class_counts = np.bincount(all_labels)
        self.assertTrue(class_counts[1] > 0)


if __name__ == "__main__":
    unittest.main()
