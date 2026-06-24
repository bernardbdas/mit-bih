import os
import wfdb
import numpy as np
from scipy import signal
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# Mapping of dataset slugs to their descriptive names on PhysioNet
DATABASES = {
    "mitdb": "MIT-BIH Arrhythmia Database",
    "pwave": "MIT-BIH Arrhythmia Database P-Wave Annotations",
    "afdb": "MIT-BIH Atrial Fibrillation Database",
    "ltdb": "MIT-BIH Long-Term ECG Database",
    "svdb": "MIT-BIH Supraventricular Arrhythmia Database",
    "stdb": "MIT-BIH ST Change Database",
    "cdb": "MIT-BIH ECG Compression Database",
    "vfdb": "MIT-BIH Malignant Ventricular Ectopy Database",
    "nstdb": "MIT-BIH Noise Stress Test Database",
    "nsrdb": "MIT-BIH Normal Sinus Rhythm Database",
    "nsr2db": "Recordings excluded from MIT-BIH Normal Sinus Rhythm DB",
    "sddb": "Sudden Cardiac Death Holter Database",
    "adfecgdb": "Abdominal and Direct Fetal ECG Database",
    "nifecgdb": "Non-Invasive Fetal ECG Arrhythmia Database",
    "slpdb": "MIT-BIH Polysomnographic Database",
    "ecg-fragment-high-risk-label": "ECG Fragment Database for the Exploration of Dangerous Arrhythmia",
    "edb": "European ST-T Database"
}

# AAMI Mapping dictionary
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal / Bundle branch block classes
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular ectopic class
    'V': 2, 'E': 2,                          # Ventricular ectopic class
    'F': 3,                                  # Fusion class
    '/': 4, 'f': 4, 'Q': 4                   # Unknown / Unclassifiable class
}

def download_database(db_slug, output_dir, overwrite=False):
    """
    Downloads a database from PhysioNet using wfdb.
    """
    db_slug = db_slug.strip().lower()
    if db_slug not in DATABASES:
        raise ValueError(f"Unknown database slug '{db_slug}'. Check DATABASES mapping.")
        
    name = DATABASES[db_slug]
    dl_path = os.path.join(output_dir, db_slug)
    os.makedirs(dl_path, exist_ok=True)
    
    print(f"Downloading: {name} ({db_slug})")
    print(f"Saving to: {os.path.abspath(dl_path)}")
    
    wfdb.dl_database(
        db_dir=db_slug,
        dl_dir=dl_path,
        overwrite=overwrite
    )
    print(f"Successfully downloaded {db_slug}!")

def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=4):
    """
    Applies a Butterworth bandpass filter to ECG data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def segment_record(record_path):
    """
    Loads an ECG record, cleans the signal via a bandpass filter,
    and extracts standardized heartbeat segments centered around annotated R-peaks.
    """
    signals, fields = wfdb.rdsamp(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    
    fs = fields['fs']
    sig = signals[:, 0]  # Use lead MLII
    
    # Clean signal
    sig_clean = bandpass_filter(sig, fs=fs)
    
    X = []
    y = []
    
    window_size = 90  # 90 samples before, 90 samples after R-peak
    
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

def process_and_segment_records(record_ids, raw_dir):
    """
    Helper to process a list of patient record IDs, segment them,
    and return stacked numpy arrays.
    """
    all_X = []
    all_y = []
    all_patient_ids = []
    
    print("Starting dataset processing and segmentation...")
    for rid in record_ids:
        r_path = os.path.join(raw_dir, rid)
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

def split_by_task(X, y, task_classes):
    """
    Split (X, y) into subsets according to task_classes list of lists.
    Returns a list of (X_t, y_t) tuples.
    """
    subsets = []
    for classes in task_classes:
        mask = np.isin(y, classes)
        subsets.append((X[mask], y[mask]))
    return subsets

def make_continual_loader(X, y, batch_size=32, shuffle=True, oversample=False):
    """
    Creates a DataLoader. If oversample=True, uses WeightedRandomSampler based on subset class frequencies.
    """
    Xt = torch.FloatTensor(X).unsqueeze(1)
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
    train_patients, test_patients, data_path,
    num_clients=3, task_classes=None, split_seed=42, batch_size=32,
    downsample_fraction=1.0
):
    """
    Partitions dataset across patients and splits by task for continual learning.
    """
    if task_classes is None:
        task_classes = [[0, 1, 2], [3, 4]]
        
    print("\n" + "="*55)
    print("  Loading data for Federated Continual Learning...")
    print("="*55)

    # Check for preprocessed files to speed up loading
    processed_path = None
    possible_processed_paths = [
        os.path.join(os.path.dirname(os.path.dirname(data_path)), "processed", "01-mit-bih-arrhythmia"),
        os.path.join(os.path.dirname(data_path), "processed", "01-mit-bih-arrhythmia"),
        os.path.abspath(os.path.join(data_path, "../processed/01-mit-bih-arrhythmia")),
        os.path.abspath(os.path.join(data_path, "../../processed/01-mit-bih-arrhythmia")),
    ]
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

    # Apply downsampling if requested
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

        # 80% train, 20% val per client
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_c, y_c, test_size=0.20,
            random_state=split_seed, stratify=y_c
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

    # Calculate class weights for reference
    counts = np.bincount(all_train_y, minlength=5)
    weights = np.sqrt(np.max(counts) / (counts + 1e-8))
    weights = np.clip(weights, 1.0, 10.0)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    return client_loaders, test_loader_full, test_loaders_by_task, class_weights

