"""
ECG signal preprocessing, segmentation, and client partitioning utilities.

This module provides routines to apply bandpass filtering, segment records around
annotated R-peaks according to the AAMI standard mapping, split datasets into
continual learning tasks, and partition patient records to simulate federated learning.
"""

import os
import glob
import wfdb
import numpy as np
from scipy import signal
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# Standard AAMI mapping for 5 Arrhythmia Classes
# N: Normal, S: Supraventricular ectopic, V: Ventricular ectopic, F: Fusion, Q: Unknown
AAMI_MAPPING: dict[str, int] = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal / Bundle branch block classes
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic class
    'V': 2, 'E': 2,                          # Ventricular ectopic class
    'F': 3,                                  # Fusion class
    '/': 4, 'f': 4, 'Q': 4                   # Unknown / Unclassifiable class
}


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 360.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to remove noise from ECG signals.

    Args:
        data: Raw 1D ECG signal array.
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        fs: Sampling frequency of the ECG signal in Hz.
        order: Order of the Butterworth filter.

    Returns:
        Filtered 1D ECG signal array.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def segment_record(record_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads an ECG record, cleans the signal via a bandpass filter,
    and extracts standardized heartbeat segments centered around annotated R-peaks.

    Heartbeats are normalized using Z-score normalization.

    Args:
        record_path: Absolute or relative path to the record (without extension).

    Returns:
        A tuple of (X, y):
            - X: 2D numpy array of shape (num_segments, 180) representing heartbeat windows.
            - y: 1D numpy array of class labels mapping to AAMI categories.
    """
    signals, fields = wfdb.rdsamp(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    
    fs = fields['fs']
    sig = signals[:, 0]  # Use lead MLII (default first column)
    
    # Clean signal using bandpass filter
    sig_clean = bandpass_filter(sig, fs=fs)
    
    X = []
    y = []
    
    window_size = 90  # 90 samples before, 90 samples after R-peak (180 total)
    
    for idx, sym in zip(ann.sample, ann.symbol):
        if sym in AAMI_MAPPING:
            if idx - window_size >= 0 and idx + window_size < len(sig_clean):
                segment = sig_clean[idx - window_size : idx + window_size]
                # Z-Score normalization
                std_val = np.std(segment)
                if std_val > 0:
                    normalized = (segment - np.mean(segment)) / std_val
                    X.append(normalized)
                    y.append(AAMI_MAPPING[sym])
                    
    return np.array(X), np.array(y)


def process_and_segment_records(
    record_ids: list[str],
    raw_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes multiple patient record IDs, segments their signals,
    and aggregates them into single numpy arrays.

    Args:
        record_ids: List of record ID strings (e.g. ['101', '106']).
        raw_dir: Base directory where raw records are located.

    Returns:
        A tuple of (X_data, y_data, patient_data):
            - X_data: 2D array of shape (N, 180) for heartbeat signals.
            - y_data: 1D array of shape (N,) for label indices.
            - patient_data: 1D array of shape (N,) matching records to patient integers.
    """
    all_X = []
    all_y = []
    all_patient_ids = []
    
    print("Starting dataset processing and segmentation...")
    for rid in record_ids:
        r_path = os.path.join(raw_dir, rid)
        # Verify the record annotation file exists
        if os.path.exists(r_path + ".atr"):
            X_rec, y_rec = segment_record(r_path)
            if len(X_rec) > 0:
                all_X.append(X_rec)
                all_y.append(y_rec)
                all_patient_ids.append(np.full(len(y_rec), int(rid)))
                
    if not all_X:
        return np.empty((0, 180)), np.empty((0,)), np.empty((0,))
        
    X_data = np.concatenate(all_X, axis=0)
    y_data = np.concatenate(all_y, axis=0)
    patient_data = np.concatenate(all_patient_ids, axis=0)
    return X_data, y_data, patient_data


def split_by_task(X: np.ndarray, y: np.ndarray, task_classes: list[list[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Splits (X, y) into subsets according to the task classes.

    Args:
        X: 2D array of ECG segments.
        y: 1D array of label indices.
        task_classes: List of class subsets representing sequential tasks
                      (e.g. [[0, 1, 2], [3, 4]]).

    Returns:
        A list of tuples, where each tuple is (X_t, y_t) for task t.
    """
    subsets = []
    for classes in task_classes:
        mask = np.isin(y, classes)
        subsets.append((X[mask], y[mask]))
    return subsets


def make_continual_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    oversample: bool = False
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for ECG segments.

    If oversample=True, uses WeightedRandomSampler based on inverse class frequencies
    to balance highly imbalanced labels.

    Args:
        X: 2D array of shape (N, 180).
        y: 1D array of labels.
        batch_size: DataLoader batch size.
        shuffle: Whether to shuffle data (ignored if oversample=True).
        oversample: Whether to balance classes using inverse frequency sampling.

    Returns:
        A configured PyTorch DataLoader.
    """
    Xt = torch.FloatTensor(X).unsqueeze(1)  # shape (N, 1, 180) for 1D CNN
    yt = torch.LongTensor(y)
    dataset = TensorDataset(Xt, yt)

    if oversample and len(y) > 0:
        present_classes, counts = np.unique(y, return_counts=True)
        class_weight_map = {c: 1.0 / cnt for c, cnt in zip(present_classes, counts)}
        sample_weights = np.array([class_weight_map[label] for label in y], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def create_federated_continual_clients(
    train_patients: list[str],
    test_patients: list[str],
    data_path: str,
    num_clients: int = 3,
    task_classes: list[list[int]] | None = None,
    split_seed: int = 42,
    batch_size: int = 32,
    downsample_fraction: float = 1.0
) -> tuple[list[dict], DataLoader, list[DataLoader | None], torch.Tensor]:
    """
    Partitions dataset across patients and splits by task for continual learning.

    Tries to load preprocessed files to save time, falling back to raw wfdb processing.

    Args:
        train_patients: Patient IDs to assign to training clients.
        test_patients: Patient IDs reserved for evaluating global models.
        data_path: Path to the raw dataset directory.
        num_clients: Number of clients to partition the training patients across.
        task_classes: Sequential lists of labels defining continual learning tasks.
        split_seed: Random seed for stratifying splits.
        batch_size: Batch size for generated loaders.
        downsample_fraction: Fraction to downsample the dataset to speed up processing/runs.

    Returns:
        A tuple of:
            - client_loaders: List of dicts, one per client, containing task loaders and metadata.
            - test_loader_full: DataLoader with the entire test set.
            - test_loaders_by_task: List of DataLoaders, one per task, for the test set.
            - class_weights: PyTorch tensor of balanced class weights based on training counts.
    """
    if task_classes is None:
        task_classes = [[0, 1, 2], [3, 4]]
        
    print("\n" + "="*55)
    print("  Loading data for Federated Continual Learning...")
    print("="*55)

    # Resolve preprocessed files to speed up loading
    processed_path = None
    project_root = os.path.abspath(os.path.join(data_path, "../../.."))
    notebooks_dir = os.path.join(project_root, "notebooks")
    
    possible_processed_paths = [os.path.abspath("assets/data")]
    if os.path.exists(notebooks_dir):
        search_pattern = os.path.join(notebooks_dir, "*", "assets", "data")
        possible_processed_paths.extend(glob.glob(search_pattern))
    for p in possible_processed_paths:
        if os.path.exists(os.path.join(p, "X.npy")):
            processed_path = p
            break

    if processed_path:
        print(f"  Found preprocessed arrays in {processed_path}. Loading in-memory...")
        X_all = np.load(os.path.join(processed_path, "X.npy"))
        y_all = np.load(os.path.join(processed_path, "y.npy"))
        patient_ids_all = np.load(os.path.join(processed_path, "patient_ids.npy"))
        
        # Filter test data
        test_p_ints = [int(p) for p in test_patients]
        test_mask = np.isin(patient_ids_all, test_p_ints)
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
    else:
        print("  Preprocessed files not found. Processing raw records (slower)...")
        X_test, y_test, _ = process_and_segment_records(test_patients, data_path)

    # Downsample if specified
    if downsample_fraction < 1.0:
        step = int(1.0 / downsample_fraction)
        X_test = X_test[::step]
        y_test = y_test[::step]
        print(f"  Downsampled test set by {downsample_fraction*100:.1f}% -> {len(y_test)} samples")

    test_loader_full = make_continual_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
    test_by_task = split_by_task(X_test, y_test, task_classes)
    test_loaders_by_task = [
        make_continual_loader(Xt, yt, batch_size=batch_size, shuffle=False) if len(yt) > 0 else None
        for (Xt, yt) in test_by_task
    ]

    print(f"  Test set: {len(y_test):,} beats (Task split: " +
          ", ".join(f"T{i+1}={len(yt)}" for i, (_, yt) in enumerate(test_by_task)) + ")")

    # Split train patients across clients
    ppc = len(train_patients) // num_clients
    client_loaders = []
    all_train_y = []

    for cid in range(num_clients):
        start = cid * ppc
        end = (start + ppc if cid < num_clients - 1 else len(train_patients))
        c_patients = train_patients[start:end]
        print(f"\n  Client {cid+1} patients: {c_patients}")

        if processed_path:
            c_p_ints = [int(p) for p in c_patients]
            c_mask = np.isin(patient_ids_all, c_p_ints)
            X_c = X_all[c_mask]
            y_c = y_all[c_mask]
        else:
            X_c, y_c, _ = process_and_segment_records(c_patients, data_path)

        if downsample_fraction < 1.0:
            step = int(1.0 / downsample_fraction)
            X_c = X_c[::step]
            y_c = y_c[::step]

        # 80% train, 20% val per client (fall back to non-stratified if any class has < 2 members)
        unique_classes, class_counts = np.unique(y_c, return_counts=True)
        can_stratify = np.all(class_counts >= 2)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_c, y_c, test_size=0.20,
            random_state=split_seed,
            stratify=y_c if can_stratify else None
        )
        all_train_y.extend(y_tr)

        train_by_task = split_by_task(X_tr, y_tr, task_classes)
        val_by_task = split_by_task(X_v, y_v, task_classes)

        train_loaders_by_task = [
            make_continual_loader(Xt, yt, batch_size=batch_size, oversample=True) if len(yt) > 0 else None
            for (Xt, yt) in train_by_task
        ]
        val_loaders_by_task = [
            make_continual_loader(Xt, yt, batch_size=batch_size, shuffle=False) if len(yt) > 0 else None
            for (Xt, yt) in val_by_task
        ]
        task_sizes = [len(yt) for (_, yt) in train_by_task]

        client_loaders.append({
            "train_by_task": train_loaders_by_task,
            "val_by_task": val_loaders_by_task,
            "task_sizes": task_sizes,
            "size": len(y_tr)
        })
        print(f"  Client {cid+1}: {len(y_tr):,} train | {len(y_v):,} val | task sizes = {task_sizes}")

    # Calculate class weights for loss function balancing
    counts = np.bincount(all_train_y, minlength=5)
    weights = np.sqrt(np.max(counts) / (counts + 1e-8))
    weights = np.clip(weights, 1.0, 10.0)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    return client_loaders, test_loader_full, test_loaders_by_task, class_weights
