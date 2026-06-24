"""
PE-FCL: Parameter-Efficient Federated Continual Learning
=========================================================
MIT-BIH Arrhythmia Database | All 7 Experiments

Experiments:
  1. FedAvg only              (baseline)
  2. LoRA only  (local)
  3. EWC only   (local)
  4. FedAvg + EWC
  5. FedAvg + LoRA
  6. EWC + LoRA (local)
  7. FedAvg + EWC + LoRA      <- YOUR METHOD (PE-FCL)

Fixes over previous version (research_baseline_v1.py -> main_pefcl.py v1):
  - REAL sequential tasks: class-incremental split.
      Task 1 = {Normal, SVEB, VEB} (labels 0,1,2)
      Task 2 = {Fusion, Unknown}   (labels 3,4)
    Each client trains on Task 1 data for the first half of rounds,
    then ONLY on Task 2 data for the second half (no replay).
    The classifier head is fixed at 5-way from the start.
  - CORRECT BWT: computed from per-task held-out test subsets,
    using accuracy measured right after a task is learned vs.
    accuracy on that same task's data after ALL tasks are done.
      BWT = mean_t [ acc(T_final, task_t) - acc(t, task_t) ]   for t < T
  - TRUE LoRA-PEFT: CNN backbone (`self.features`) is fully frozen
    when use_lora=True. Only LoRA A/B matrices (+ small classifier
    head when LoRA is off) are trainable.
  - Multi-seed protocol: runs N_SEEDS independent repetitions of
    the full 7-experiment suite and reports mean +/- std per metric.
  - Kept from v1: weighted FedAvg aggregation, macro-F1, communication
    cost tracking, weights_only=True on torch.load (n/a here, no
    checkpoint loading in this script, kept for any future use),
    inter-patient train/test split.

Run: python main_pefcl.py
Expected time: scales with N_SEEDS; budget ~1-2 hours per seed on CPU.
"""

import os
import random
import copy
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, TensorDataset, WeightedRandomSampler)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"Running on: {DEVICE}")

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(SCRIPT_DIR,
                "mit-bih-arrhythmia-database-1.0.0")

NUM_CLIENTS   = 3
NUM_ROUNDS    = 20          # split evenly across tasks (must be even)
                              # raised from 10 -> 20 for more thorough
                              # per-task training (was likely undertrained)
LOCAL_EPOCHS  = 10           # raised from 3 -> 10; PATIENCE below still
                              # caps any single round's local training if
                              # val loss stops improving, so this raises
                              # the ceiling without removing the safety net
BACKBONE_PRETRAIN_ROUNDS = 5   # federated rounds used to pretrain the CNN
                                 # backbone on Task 1 data ONLY, before it is
                                 # frozen for LoRA experiments (2, 5, 6, 7).
                                 # Without this, a frozen RANDOM backbone gives
                                 # LoRA nothing useful to adapt on top of, and
                                 # training collapses to majority-class
                                 # prediction regardless of seed (see notes on
                                 # load_pretrained_backbone). This phase still
                                 # respects FL constraints: each client trains
                                 # locally on its own Task 1 data only, and
                                 # only model weights (not data) are aggregated
                                 # -- exactly like a normal FedAvg round, just
                                 # run before the experiment's main rounds and
                                 # using the un-frozen, non-LoRA architecture.
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
WINDOW_SIZE   = 90      # each side of R-peak -> 180 samples total
PATIENCE      = 5       # early stopping patience
DROPOUT_RATE  = 0.3     # lowered from 0.5; 0.5 risks underfitting on a
                          # CNN this size for ECG beat classification
LORA_RANK     = 8       # raised from 4; rank 4 may be too restrictive
                          # now that the backbone is pretrained and the
                          # LoRA head has meaningful features to adapt
LORA_ALPHA    = 1.0     # LoRA scaling
EWC_LAMBDA_FULL = 50.0    # lowered from 400.0; 400 was tuned before Fisher
                            # normalisation was added and was producing
                            # severe forgetting (BWT ~ -88), suggesting the
                            # penalty was either dominating in a way that
                            # destabilised training, or simply mis-scaled
                            # for the normalised-Fisher scheme now in use.
                            # NOTE: training loss is now Focal Loss (see
                            # FocalLoss class), which is typically SMALLER
                            # in magnitude than CrossEntropyLoss once
                            # predictions are reasonably confident. If the
                            # "EWC penalty >> task loss" warning in
                            # train_local fires often, lower this further.
                            # (use_lora=False: methods 1,3,4)
EWC_LAMBDA_LORA = 1.0      # lowered from 5.0, same reasoning, scaled down
                            # further since LoRA's parameter set is tiny
                            # and the backbone is now pretrained+frozen
                            # rather than random, so the Fisher signal is
                            # more meaningful and needs less damping.

# ---- Multi-seed protocol ----
SEEDS = [42, 0, 1, 2, 3]    # 5 seeds; trim to [42, 0, 1] for a quick run

# ---- Sequential task definition (class-incremental) ----
# Task 1: common classes. Task 2: rare classes (introduced later,
# never replayed). Head stays 5-way throughout (fixed-head CIL).
TASK_CLASSES = [
    [0, 1, 2],   # Task 1: Normal, SVEB, VEB
    [3, 4],      # Task 2: Fusion, Unknown
]
NUM_TASKS = len(TASK_CLASSES)
assert NUM_ROUNDS % NUM_TASKS == 0, \
    "NUM_ROUNDS must split evenly across tasks"
ROUNDS_PER_TASK = NUM_ROUNDS // NUM_TASKS

# ============================================================
# AAMI 5-CLASS MAPPING
# ============================================================
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # SVEB
    'V': 2, 'E': 2,                               # VEB
    'F': 3,                                        # Fusion
    '/': 4, 'f': 4, 'Q': 4                        # Unknown
}

CLASS_NAMES = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]

# ============================================================
# INTER-PATIENT SPLIT
# Standard in MIT-BIH literature -- prevents data leakage
# ============================================================
TRAIN_PATIENTS = [
    '101', '106', '108', '109', '112', '114',
    '115', '116', '118', '119', '122', '124',
    '201', '203', '205', '207', '208', '209',
    '215', '220', '223', '230'
]

TEST_PATIENTS = [
    '100', '103', '104', '105', '111', '113',
    '117', '121', '123', '200', '210', '212',
    '213', '214', '217', '219', '221', '222',
    '228', '231', '232', '233', '234'
]


# ============================================================
# REPRODUCIBILITY HELPERS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PART 1 -- DATA LOADING
# ============================================================
def extract_beats(patient_ids):
    """
    Extract heartbeat segments from MIT-BIH records.
    Each beat: WINDOW_SIZE samples each side of R-peak.
    Z-score normalised per beat.
    """
    X_all, y_all = [], []

    for pid in patient_ids:
        record_path = os.path.join(DATA_PATH, pid)
        if not os.path.exists(record_path + ".dat"):
            print(f"  Skipping missing record: {pid}")
            continue

        import wfdb
        record     = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
        signal     = record.p_signal[:, 0]

        for peak, symbol in zip(annotation.sample,
                                annotation.symbol):
            if symbol not in AAMI_MAPPING:
                continue
            s = peak - WINDOW_SIZE
            e = peak + WINDOW_SIZE
            if s < 0 or e >= len(signal):
                continue

            beat     = signal[s:e].copy()
            std_val  = np.std(beat)
            beat     = ((beat - np.mean(beat)) /
                        (std_val if std_val > 0 else 1e-8))
            X_all.append(beat.astype(np.float32))
            y_all.append(AAMI_MAPPING[symbol])

        print(f"  OK {pid}: {len(y_all)} total beats")

    return np.array(X_all, dtype=np.float32), \
           np.array(y_all,  dtype=np.int64)


def calculate_class_weights(labels):
    """
    Compute inverse-frequency class weights.
    Uses sqrt to avoid extreme values for very rare classes.
    """
    counts  = np.bincount(labels, minlength=5)
    weights = np.sqrt(np.max(counts) / (counts + 1e-8))
    weights = np.clip(weights, 1.0, 10.0)
    print(f"\n  Class weights: {np.round(weights, 2)}")
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(X, y, shuffle=True, oversample=False):
    """
    oversample=True uses a WeightedRandomSampler (inverse-frequency,
    computed from the classes ACTUALLY PRESENT in this particular
    X/y, e.g. just Task 1's 3 classes or just Task 2's 2 classes --
    not the global 5-class distribution). This matters: applying a
    sampler built from global class frequencies to a task-filtered
    subset would weight by classes that aren't even present in that
    subset, which is meaningless. Rebalancing within the classes a
    given task/client loader actually contains is what helps rare
    classes (e.g. Fusion within Task 2, where it competes only
    against Unknown, not against Normal/SVEB/VEB it'll never see).

    oversample is incompatible with shuffle (the sampler replaces
    shuffling) and is only meant for TRAINING loaders, never val/test.
    """
    Xt = torch.FloatTensor(X).unsqueeze(1)
    yt = torch.LongTensor(y)
    dataset = TensorDataset(Xt, yt)

    if oversample and len(y) > 0:
        present_classes, counts = np.unique(y, return_counts=True)
        class_weight_map = {
            c: 1.0 / cnt for c, cnt in zip(present_classes, counts)
        }
        sample_weights = np.array(
            [class_weight_map[label] for label in y], dtype=np.float64)
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True)
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)

    return DataLoader(dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=shuffle,
                      num_workers=0)


def split_by_task(X, y):
    """
    Split (X, y) into NUM_TASKS subsets according to TASK_CLASSES.
    Returns a list of (X_t, y_t) tuples, one per task, preserving
    only the samples whose label belongs to that task's class set.
    """
    subsets = []
    for classes in TASK_CLASSES:
        mask = np.isin(y, classes)
        subsets.append((X[mask], y[mask]))
    return subsets


def create_federated_clients(split_seed):
    """
    Load data with proper train / val / test separation, and split
    every train/val/test pool further into per-task subsets for
    class-incremental continual learning.

    TEST SET   -> held-out patients (never used during training)
    TRAIN SET  -> split across NUM_CLIENTS
    VAL SET    -> 20% of each client's data

    Each of train/val/test is additionally split into NUM_TASKS
    task-specific subsets (by class membership) so that:
      - clients can train sequentially, task by task, with no replay
      - BWT can be computed against per-task held-out test data

    split_seed: fixes the train/val split so all seeds in the
    multi-seed protocol see the EXACT same data partition; only
    model initialisation / training stochasticity varies by seed.
    """
    print("\n" + "="*55)
    print("  Loading data...")
    print("="*55)

    # -- Load held-out test set ---------------------------------
    print("\nTest patients (held-out):")
    X_test, y_test = extract_beats(TEST_PATIENTS)
    test_loader_full = make_loader(X_test, y_test, shuffle=False)
    test_by_task = split_by_task(X_test, y_test)
    test_loaders_by_task = [
        make_loader(Xt, yt, shuffle=False) if len(yt) > 0 else None
        for (Xt, yt) in test_by_task
    ]
    print(f"  Test set: {len(y_test):,} beats "
          f"(Task split: " +
          ", ".join(f"T{i+1}={len(yt)}"
                     for i, (_, yt) in enumerate(test_by_task)) +
          ")")

    # -- Create federated clients from train patients ------------
    print("\nTrain patients -> split across clients:")
    ppc            = len(TRAIN_PATIENTS) // NUM_CLIENTS
    client_loaders = []
    all_train_y    = []

    for cid in range(NUM_CLIENTS):
        start = cid * ppc
        end   = (start + ppc if cid < NUM_CLIENTS - 1
                 else len(TRAIN_PATIENTS))
        c_patients = TRAIN_PATIENTS[start:end]
        print(f"\n  Client {cid+1} patients: {c_patients}")

        X_c, y_c = extract_beats(c_patients)

        # 80% train, 20% val -- per client (stratified overall)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_c, y_c, test_size=0.20,
            random_state=split_seed, stratify=y_c)

        all_train_y.extend(y_tr)

        # Per-task subsets, train and val
        train_by_task = split_by_task(X_tr, y_tr)
        val_by_task   = split_by_task(X_v, y_v)

        train_loaders_by_task = [
            make_loader(Xt, yt, oversample=True) if len(yt) > 0 else None
            for (Xt, yt) in train_by_task
        ]
        val_loaders_by_task = [
            make_loader(Xt, yt, shuffle=False) if len(yt) > 0 else None
            for (Xt, yt) in val_by_task
        ]
        task_sizes = [len(yt) for (_, yt) in train_by_task]

        client_loaders.append({
            "train_by_task": train_loaders_by_task,
            "val_by_task"  : val_loaders_by_task,
            "task_sizes"   : task_sizes,
            # full-client loaders kept for non-CL ablations if ever
            # needed, and for overall size accounting
            "size": len(y_tr)
        })
        print(f"  Client {cid+1}: {len(y_tr):,} train | "
              f"{len(y_v):,} val | "
              f"task sizes (train) = {task_sizes}")

    class_weights = calculate_class_weights(
        np.array(all_train_y))

    print(f"\n  Summary:")
    print(f"  Test set    : {len(y_test):,} beats "
          f"(held-out patients -- never trained on)")
    for cid, cl in enumerate(client_loaders):
        print(f"  Client {cid+1}   : {cl['size']:,} train beats")

    return (client_loaders, test_loader_full, test_loaders_by_task,
            class_weights)


# ============================================================
# PART 2 -- MODEL ARCHITECTURE
# ============================================================
class LoRALayer(nn.Module):
    """
    Linear layer with LoRA adapters.

    W0 is FROZEN (pre-trained / randomly-initialised-then-frozen
    base weights). Only A and B are trained and transmitted to
    the server.
    Output = W0(x) + alpha * B(A(x))

    Communication saving:
      Full layer: in_features x out_features params sent
      LoRA:       (in_features + out_features) x rank params sent
    """
    def __init__(self, in_features, out_features,
                 rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        # Frozen base weights
        self.W0 = nn.Linear(in_features, out_features, bias=True)
        self.W0.weight.requires_grad = False
        self.W0.bias.requires_grad   = False

        # Trainable low-rank matrices
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # Init: A random, B zero -> LoRA contribution starts at 0
        nn.init.kaiming_uniform_(self.A.weight,
                                 a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.W0(x) + self.alpha * self.B(self.A(x))


class ECGCNN(nn.Module):
    """
    1D-CNN for ECG beat classification.
    Input : (batch, 1, 180) -- 180-sample heartbeat window
    Output: (batch, 5)      -- 5 arrhythmia classes (fixed head,
                                class-incremental over time)

    use_lora=True:
      - CNN backbone (`self.features`) is FULLY FROZEN.
      - fc1 (256->64) is FROZEN too (it is a fixed projector).
      - Only the final LoRA-adapted classifier (A, B matrices) is
        trainable. This is true LoRA-based PEFT: the vast majority
        of parameters never move, and only a thin low-rank slice
        is trained and exchanged with the server.
    use_lora=False:
      - Standard fully fine-tuned CNN (all params trainable). This
        is the non-PEFT baseline against which LoRA communication
        and forgetting trade-offs are measured.
    """
    def __init__(self, use_lora=False):
        super().__init__()
        self.use_lora = use_lora

        self.features = nn.Sequential(
            nn.Conv1d(1,  64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        if use_lora:
            self.fc1        = nn.Linear(256, 64)
            self.relu       = nn.ReLU()
            self.dropout    = nn.Dropout(DROPOUT_RATE)
            self.classifier = LoRALayer(64, 5,
                                        rank=LORA_RANK,
                                        alpha=LORA_ALPHA)

            # --- TRUE PEFT: freeze everything except LoRA A/B ---
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.fc1.parameters():
                p.requires_grad = False
            # LoRALayer already freezes its own W0/bias internally.
            # BatchNorm running stats still update during forward()
            # in train() mode (this is expected/standard even when
            # affine params are frozen-by-extension here since BN1d
            # has its own weight/bias which we also freeze below).
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(64, 5)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)    # flatten
        if self.use_lora:
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def load_pretrained_backbone(self, pretrained_model):
        """
        Copy `features` (the CNN backbone) and the 256->64 projector
        from a PRETRAINED non-LoRA ECGCNN into this LoRA model.

        This only makes sense to call on a model with use_lora=True,
        loading from a model with use_lora=False that was already
        trained (e.g. on Task 1 data) so its backbone produces
        class-discriminative features rather than random ones.

        Without this step, a frozen-from-random-init backbone carries
        essentially no useful signal, and a rank-r LoRA head trained
        on top of it cannot recover a working classifier -- training
        collapses to predicting the majority class regardless of
        seed or hyperparameters (this is what produced the 81.91%
        acc / 20.00% recall / identical-across-seeds result: that
        is the exact macro-recall signature of an always-predict-
        majority-class model).

        pretrained_model.classifier is expected to be the standard
        nn.Sequential(Flatten, Linear(256,64), ReLU, Dropout, Linear(64,5))
        used when use_lora=False; its Linear(256,64) becomes this
        model's fc1, and its final Linear(64,5) weights seed this
        model's LoRA W0 (frozen base) so the LoRA branch starts from
        a sensible decision boundary too, not just a random one.
        """
        if not self.use_lora:
            raise ValueError(
                "load_pretrained_backbone only applies to "
                "use_lora=True models")
        if pretrained_model.use_lora:
            raise ValueError(
                "pretrained_model must be a use_lora=False model "
                "(the one actually trained on real data)")

        self.features.load_state_dict(
            pretrained_model.features.state_dict())

        # pretrained_model.classifier = Sequential(Flatten, Linear(256,64),
        #                                           ReLU, Dropout, Linear(64,5))
        pretrained_fc1   = pretrained_model.classifier[1]
        pretrained_final = pretrained_model.classifier[4]

        self.fc1.load_state_dict(pretrained_fc1.state_dict())

        # Seed the LoRA base layer's frozen W0 with the pretrained
        # final classifier weights, so W0(x) alone is already a
        # reasonable Task-1 classifier before any LoRA adaptation.
        self.classifier.W0.weight.data = (
            pretrained_final.weight.data.clone())
        self.classifier.W0.bias.data = (
            pretrained_final.bias.data.clone())

        # Re-affirm frozen status (load_state_dict / direct .data
        # assignment do not change requires_grad, but keep this
        # explicit and defensive).
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.fc1.parameters():
            p.requires_grad = False
        self.classifier.W0.weight.requires_grad = False
        self.classifier.W0.bias.requires_grad = False

    def get_lora_params(self):
        """Return only LoRA A and B weights for transmission."""
        if not self.use_lora:
            raise ValueError("Model not using LoRA")
        return {
            'A': self.classifier.A.weight.data.clone(),
            'B': self.classifier.B.weight.data.clone()
        }

    def set_lora_params(self, params):
        """Load aggregated LoRA weights from server."""
        self.classifier.A.weight.data = params['A'].clone()
        self.classifier.B.weight.data = params['B'].clone()

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_train_mode(self):
        """
        Put the model in the correct train/eval mode given what is
        actually trainable.

        When use_lora=True the backbone (`features`) and `fc1` are
        frozen. A plain `model.train()` would still let BatchNorm1d
        layers in `features` update their running_mean/running_var
        on every forward pass even though weight/bias don't get
        gradients -- that's "frozen" in name only. To get a TRULY
        frozen backbone (the point of LoRA-based PEFT), those
        BatchNorm layers must be kept in eval() mode (so running
        stats stay fixed at whatever they were at initialisation /
        loaded from), while the trainable parts (fc1->relu->dropout
        is frozen too here, only the LoRA classifier trains) stay
        in train() mode for things like Dropout to behave correctly.
        """
        if self.use_lora:
            self.eval()                 # freeze everything by default...
            self.classifier.train()     # ...then re-enable train mode
            self.dropout.train()        # for the trainable LoRA head
                                          # and its preceding Dropout,
                                          # so Dropout actually drops
                                          # units during local training
        else:
            self.train()


# ============================================================
# PART 3 -- EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience  = patience
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# PART 2b -- FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017), used here INSTEAD OF class-weighted
    CrossEntropyLoss for training. Down-weights easy/well-classified
    examples and focuses gradient on hard/rare ones, adaptively --
    without a fixed per-class multiplier.

    Deliberately used UNWEIGHTED (no class weight tensor) because
    training data is already rebalanced per-task via a
    WeightedRandomSampler (see make_loader(oversample=True)).
    Stacking inverse-frequency class weights, a rebalancing sampler,
    AND focal loss all at once over-corrects for rare classes and
    was part of what caused an earlier collapse where the model
    predicted only the rarest class (Fusion) for everything. Pick
    ONE rebalancing mechanism's worth of strength, not three.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(
            inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================
# PART 4 -- EWC
# ============================================================
def compute_fisher(model, dataloader, criterion):
    """
    Compute diagonal Fisher Information Matrix over TRAINABLE
    parameters only.

    Fisher score F_i for parameter theta_i answers:
    "How much does the current task's loss change if I perturb
    theta_i?" High F_i -> parameter is important -> protect it
    when training on the next task.

    Because the backbone is frozen when use_lora=True, this Fisher
    is computed only over the LoRA A/B matrices in that setting,
    which is what gives PE-FCL its large reduction in EWC bookkeeping
    cost (no need to store/penalise a full-model Fisher matrix).
    """
    if dataloader is None:
        return {}, {}

    fisher  = {}
    opt_par = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name]  = torch.zeros_like(param.data)
            opt_par[name] = param.data.clone()

    model.eval()
    n_batches = len(dataloader)
    if n_batches == 0:
        return fisher, opt_par

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        model.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += (param.grad.data.pow(2)
                                 / n_batches)

    return fisher, opt_par


def ewc_penalty(model, fisher, opt_par):
    """
    EWC penalty added to the local loss while training on the
    CURRENT task, anchored to the parameter values recorded right
    after the PREVIOUS task finished.

    L_total = L_current_task
              + (lambda/2) * sum_i F_i_norm * (theta_i - theta*_i)^2

    Parameters with high Fisher scores (i.e. that mattered for the
    previous task) are penalised heavily for moving away from their
    post-previous-task values -- this is what preserves old-task
    knowledge while learning the new task.

    Two stabilisation choices, both motivated by an earlier failure
    mode where FedAvg+EWC+LoRA collapsed to predicting a single class
    (degenerate ~6% accuracy, one-hot confusion matrix):

    1. Lambda depends on whether the backbone is frozen (LoRA) or not.
       A full CNN and a rank-4 LoRA head have wildly different parameter
       counts and Fisher magnitudes; reusing one lambda for both lets
       the penalty dominate the task loss on the smaller LoRA head.
    2. Fisher values are normalised by their own max (per call) before
       being used as weights. Raw Fisher values are unbounded and can
       grow across EWC's running-sum accumulation over tasks; without
       normalisation the penalty term can explode in scale over time
       regardless of which lambda is chosen.
    """
    if not fisher:
        return torch.tensor(0.0, device=DEVICE)

    ewc_lambda = EWC_LAMBDA_LORA if model.use_lora else EWC_LAMBDA_FULL

    # Normalise Fisher so the penalty's overall scale stays bounded
    # no matter how many parameters are trainable or how many tasks'
    # worth of Fisher have been accumulated.
    max_fisher = max(
        (f.max().item() for f in fisher.values() if f.numel() > 0),
        default=0.0
    )
    norm = max_fisher if max_fisher > 1e-12 else 1.0

    loss = torch.tensor(0.0, device=DEVICE)
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher:
            loss += (
                (fisher[name].to(DEVICE) / norm) *
                (param - opt_par[name].to(DEVICE)).pow(2)
            ).sum()
    return (ewc_lambda / 2) * loss


# ============================================================
# PART 5 -- TRAINING AND EVALUATION HELPERS
# ============================================================
def validate_model(model, dataloader, criterion):
    if dataloader is None or len(dataloader) == 0:
        return 0.0, 0.0
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs     = model(inputs)
            total_loss += criterion(outputs, labels).item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / len(dataloader), 100 * correct / total


def evaluate_model(model, dataloader):
    """
    Full evaluation on a (test) loader.
    Reports accuracy, macro F1, precision, recall.
    Uses macro averaging -- correct for imbalanced datasets.
    Returns None if the loader is empty (e.g. a per-task test
    loader for a task with no held-out samples).
    """
    if dataloader is None or len(dataloader) == 0:
        return None

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    if len(y_true) == 0:
        return None

    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        # NOTE: macro averaging used (not weighted)
        # This is the correct metric for imbalanced medical data
        "precision": precision_score(y_true, y_pred,
                                     average="macro",
                                     zero_division=0),
        "recall"   : recall_score(y_true, y_pred,
                                  average="macro",
                                  zero_division=0),
        "f1"       : f1_score(y_true, y_pred,
                              average="macro",
                              zero_division=0),
        "cm"       : confusion_matrix(y_true, y_pred,
                                       labels=list(range(5))),
        "preds"    : y_pred,
        "true"     : y_true
    }


def train_local(model, train_loader, val_loader,
                criterion, use_ewc=False,
                fisher=None, opt_par=None):
    """
    Train one client locally for LOCAL_EPOCHS epochs on the
    CURRENT task's data only. Includes early stopping and optional
    EWC penalty anchored to the previous task's parameters.
    Returns best model (lowest val loss seen during local training).

    If train_loader is None or empty (task has no data for this
    client), the model is returned unchanged.
    """
    if train_loader is None or len(train_loader) == 0:
        return model

    model     = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4)
    # Decays LR when val loss plateaus -- helps stabilise convergence
    # now that LOCAL_EPOCHS is higher (10) and a single local training
    # call has more room to drift without a correction signal.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    early_stopper = EarlyStopping(patience=PATIENCE)
    best_state    = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(LOCAL_EPOCHS):
        model.set_train_mode()
        running_loss      = 0.0
        running_task_loss = 0.0
        running_ewc_loss  = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            task_loss = criterion(model(inputs), labels)
            loss = task_loss

            # Add EWC penalty if Fisher computed from a prior task
            ewc_loss = torch.tensor(0.0, device=DEVICE)
            if use_ewc and fisher:
                ewc_loss = ewc_penalty(model, fisher, opt_par)
                loss = loss + ewc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0)
            optimizer.step()
            running_loss      += loss.item()
            running_task_loss += task_loss.item()
            running_ewc_loss  += ewc_loss.item()

        train_loss = running_loss / len(train_loader)
        avg_task_loss = running_task_loss / len(train_loader)
        avg_ewc_loss  = running_ewc_loss / len(train_loader)
        val_loss, val_acc = validate_model(
            model, val_loader, criterion)
        scheduler.step(val_loss)

        # Flag if EWC is dominating the loss -- this is the exact
        # signature of the collapse seen previously (EWC penalty
        # swamping task loss and driving the model to a degenerate
        # one-class solution). Tune EWC_LAMBDA_LORA / EWC_LAMBDA_FULL
        # down if this fires repeatedly.
        if use_ewc and avg_task_loss > 1e-8 and \
                avg_ewc_loss > 5 * avg_task_loss:
            print(f"      WARNING: EWC penalty ({avg_ewc_loss:.4f}) "
                  f">> task loss ({avg_task_loss:.4f}). EWC may be "
                  f"dominating training -- consider lowering "
                  f"{'EWC_LAMBDA_LORA' if model.use_lora else 'EWC_LAMBDA_FULL'}.")

        print(f"      Epoch {epoch+1}/{LOCAL_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} "
              f"(task={avg_task_loss:.4f} ewc={avg_ewc_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())

        early_stopper(val_loss)
        if early_stopper.stop:
            print("      Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    return model


# ============================================================
# PART 6 -- FEDAVG AGGREGATION
# ============================================================
def federated_average(local_weights, client_sizes):
    """
    Weighted FedAvg aggregation.
    Larger clients contribute proportionally more.
    Clients with zero data for the current task (size 0) are
    excluded from the weighting so they don't dilute the average
    with their unchanged/stale weights.
    """
    weights_and_sizes = [
        (w, s) for w, s in zip(local_weights, client_sizes) if s > 0
    ]
    if not weights_and_sizes:
        # no client had data this round; nothing to aggregate
        return local_weights[0]

    total = sum(s for _, s in weights_and_sizes)
    agg   = {}
    for key in weights_and_sizes[0][0].keys():
        agg[key] = sum(
            (s / total) * w[key].float()
            for w, s in weights_and_sizes
        )
    return agg


def get_weights_to_send(model, use_lora):
    """
    Get weights that are transmitted to server.
    LoRA: only A and B matrices (~96% less data)
    Full: entire model state dict
    """
    if use_lora:
        lp = model.get_lora_params()
        return {'A': lp['A'], 'B': lp['B']}
    else:
        return copy.deepcopy(model.state_dict())


def count_params_sent(weights):
    return sum(v.numel() for v in weights.values())


# ============================================================
# PART 7 -- BWT METRIC (Catastrophic Forgetting)
# ============================================================
def compute_bwt(per_task_acc_when_learned, per_task_acc_final):
    """
    Backward Transfer (BWT), standard continual-learning definition
    (Lopez-Paz & Ranzato, 2017):

        BWT = 1/(T-1) * sum_{t=1}^{T-1} [ A_{T,t} - A_{t,t} ]

    where:
      A_{t,t}  = accuracy on task t's held-out data measured right
                 after task t was learned (per_task_acc_when_learned[t])
      A_{T,t}  = accuracy on task t's held-out data measured at the
                 very end, after the LAST task T was learned
                 (per_task_acc_final[t])

    Only tasks t = 1 .. T-1 are included (the final task T has no
    "later" point to measure forgetting against).

    Positive BWT -> model improved on old tasks (positive transfer)
    Negative BWT -> model forgot old tasks (catastrophic forgetting)
    Near zero    -> stable knowledge retention

    Returns None if there are fewer than 2 tasks with valid scores.
    """
    diffs = []
    for t in range(NUM_TASKS - 1):
        a_tt = per_task_acc_when_learned[t]
        a_Tt = per_task_acc_final[t]
        if a_tt is None or a_Tt is None:
            continue
        diffs.append(a_Tt - a_tt)

    if not diffs:
        return None
    return float(np.mean(diffs))


# ============================================================
# PART 7b -- FEDERATED BACKBONE PRETRAINING (for LoRA experiments)
# ============================================================
def pretrain_backbone(use_fedavg, client_loaders, criterion):
    """
    Federated pretraining of the CNN backbone on TASK 1 data only,
    using a temporary non-LoRA (fully trainable) ECGCNN. Returns the
    trained model, whose `features` + first classifier Linear(256,64)
    will be transplanted into the LoRA model and then frozen.

    This is required because LoRA adapts an existing, meaningful
    representation -- it cannot learn one from scratch through a
    rank-r bottleneck on top of a frozen RANDOM backbone. Without
    this step, LoRA experiments collapse to predicting the majority
    class regardless of seed (see ECGCNN.load_pretrained_backbone).

    Stays within FL constraints: each client trains locally on its
    own Task 1 data only; only model weights are aggregated across
    clients (use_fedavg controls whether that aggregation happens,
    matching the same flag used in the main experiment loop -- so
    "local only" experiments also pretrain locally only, with no
    cross-client weight sharing, for consistency with their ablation).
    """
    print(f"\n  {'-'*51}")
    print(f"  Pretraining backbone on Task 1 data "
          f"({BACKBONE_PRETRAIN_ROUNDS} rounds, fedavg={use_fedavg})")
    print(f"  {'-'*51}")

    pretrain_model = ECGCNN(use_lora=False).to(DEVICE)
    client_models = [copy.deepcopy(pretrain_model)
                     for _ in range(NUM_CLIENTS)]
    client_sizes_task1 = [
        cl['task_sizes'][0] for cl in client_loaders
    ]

    for rnd in range(BACKBONE_PRETRAIN_ROUNDS):
        local_weights = []
        for cid in range(NUM_CLIENTS):
            train_loader = client_loaders[cid]['train_by_task'][0]
            val_loader   = client_loaders[cid]['val_by_task'][0]
            client_models[cid] = train_local(
                client_models[cid], train_loader, val_loader,
                criterion, use_ewc=False)
            local_weights.append(
                copy.deepcopy(client_models[cid].state_dict()))

        if use_fedavg:
            agg = federated_average(local_weights, client_sizes_task1)
            pretrain_model.load_state_dict(agg)
            for cid in range(NUM_CLIENTS):
                client_models[cid].load_state_dict(copy.deepcopy(agg))
        else:
            pretrain_model = copy.deepcopy(client_models[0])

        _, val_acc = validate_model(
            pretrain_model, client_loaders[0]['val_by_task'][0],
            criterion)
        print(f"    Pretrain round {rnd+1}/{BACKBONE_PRETRAIN_ROUNDS} "
              f"| client0 val acc: {val_acc:.2f}%")

    print(f"  Backbone pretraining done.\n")
    return pretrain_model


# ============================================================
# PART 8 -- MAIN EXPERIMENT RUNNER
# ============================================================
def run_experiment(name, use_fedavg, use_ewc, use_lora,
                   client_loaders, test_loader_full,
                   test_loaders_by_task, class_weights, seed):
    """
    Run one complete federated CONTINUAL-LEARNING experiment.

    Training proceeds task by task (class-incremental, no replay):
      - rounds 0 .. ROUNDS_PER_TASK-1            -> Task 1 only
      - rounds ROUNDS_PER_TASK .. NUM_ROUNDS-1    -> Task 2 only
    (generalises to NUM_TASKS > 2 the same way)

    Tracks:
      - Val accuracy per round
      - Communication cost per round
      - Per-task test accuracy snapshot right when each task ends
        (A_{t,t}) and again at the very end (A_{T,t}) -> used for BWT
      - Final test accuracy / F1 / precision / recall on the FULL
        held-out test set (all classes / all tasks combined)
    """
    print(f"\n{'='*55}")
    print(f"  Experiment: {name}  (seed={seed})")
    print(f"  fedavg={use_fedavg} | "
          f"ewc={use_ewc} | lora={use_lora}")
    print(f"{'='*55}")

    # Focal loss (unweighted) replaces class-weighted CrossEntropyLoss
    # for TRAINING. Training data is already rebalanced per-task via
    # WeightedRandomSampler (see create_federated_clients/make_loader),
    # so an additional fixed inverse-frequency weight here would
    # double-correct for rare classes -- this combination previously
    # caused the model to collapse to predicting only the rarest class.
    # class_weights is still computed and returned for reference /
    # any future use, just not plugged into the training loss anymore.
    criterion = FocalLoss(gamma=2.0)

    # Build model
    global_model = ECGCNN(use_lora=use_lora).to(DEVICE)

    if use_lora:
        # Pretrain a non-LoRA backbone on Task 1 data, then transplant
        # its weights into this LoRA model before freezing them. A
        # frozen RANDOM backbone gives LoRA nothing useful to adapt;
        # see pretrain_backbone() / load_pretrained_backbone() docs.
        pretrained = pretrain_backbone(
            use_fedavg, client_loaders, criterion)
        global_model.load_pretrained_backbone(pretrained)

    print(f"  Trainable params: "
          f"{global_model.trainable_params():,} / "
          f"{global_model.total_params():,} "
          f"({global_model.trainable_params()/global_model.total_params()*100:.1f}%)")

    # EWC state per client: Fisher/opt_par recorded after the most
    # recently completed task (None until task 1 finishes)
    client_fisher = [None] * NUM_CLIENTS
    client_optpar = [None] * NUM_CLIENTS

    # Track per-round metrics
    round_val_accs  = []
    round_comm_cost = []

    # Per-task bookkeeping for BWT
    per_task_acc_when_learned = [None] * NUM_TASKS

    # Initialise client models
    client_models = [copy.deepcopy(global_model)
                     for _ in range(NUM_CLIENTS)]

    for rnd in range(NUM_ROUNDS):
        task = rnd // ROUNDS_PER_TASK
        is_last_round_of_task = (rnd + 1) % ROUNDS_PER_TASK == 0
        print(f"\n  Round {rnd+1}/{NUM_ROUNDS} | Task {task+1}/{NUM_TASKS}")

        local_weights = []
        round_comm    = 0
        round_val_sum = 0.0
        round_val_n   = 0

        # Per-task client sizes (for weighted FedAvg THIS round)
        client_sizes_this_task = [
            cl['task_sizes'][task] for cl in client_loaders
        ]
        print(f"  Client sizes for Task {task+1}: "
              f"{client_sizes_this_task}")
        if sum(1 for s in client_sizes_this_task if s > 0) <= 1:
            print(f"  WARNING: {sum(1 for s in client_sizes_this_task if s == 0)} "
                  f"/ {NUM_CLIENTS} clients have ZERO data for this task. "
                  f"FedAvg effectively reduces to single-client training "
                  f"this round -- results may not reflect genuine "
                  f"federated behaviour.")

        for cid in range(NUM_CLIENTS):
            print(f"\n    Client {cid+1}")

            train_loader = client_loaders[cid]['train_by_task'][task]
            val_loader   = client_loaders[cid]['val_by_task'][task]

            # Train locally on CURRENT task's data only
            client_models[cid] = train_local(
                client_models[cid],
                train_loader,
                val_loader,
                criterion,
                use_ewc=use_ewc,
                fisher=client_fisher[cid],
                opt_par=client_optpar[cid]
            )

            # Get weights to send to server
            w = get_weights_to_send(
                client_models[cid], use_lora)
            local_weights.append(w)
            round_comm += count_params_sent(w)

            # Val accuracy for this client on its current task
            if val_loader is not None and len(val_loader) > 0:
                _, val_acc = validate_model(
                    client_models[cid], val_loader, criterion)
                round_val_sum += val_acc
                round_val_n   += 1

        # Server aggregation
        if use_fedavg:
            agg = federated_average(
                local_weights, client_sizes_this_task)

            if use_lora:
                for cid in range(NUM_CLIENTS):
                    client_models[cid].set_lora_params({
                        'A': agg['A'], 'B': agg['B']})
                global_model.set_lora_params({
                    'A': agg['A'], 'B': agg['B']})
            else:
                global_model.load_state_dict(agg)
                for cid in range(NUM_CLIENTS):
                    client_models[cid].load_state_dict(
                        copy.deepcopy(agg))
        else:
            # No federation -- use client 0 as representative
            global_model = copy.deepcopy(client_models[0])

        avg_val = (round_val_sum / round_val_n) if round_val_n else 0.0
        round_val_accs.append(avg_val)
        round_comm_cost.append(round_comm)

        print(f"\n  Round {rnd+1} avg val acc: "
              f"{avg_val:.2f}% | "
              f"Comm params: {round_comm:,}")

        # ---- End of task: compute Fisher AFTER training on it ----
        if is_last_round_of_task:
            if use_ewc:
                for cid in range(NUM_CLIENTS):
                    fisher, opt_par = compute_fisher(
                        client_models[cid],
                        client_loaders[cid]['train_by_task'][task],
                        criterion)
                    # Merge with previous Fisher (simple running sum,
                    # standard online-EWC style accumulation) so that
                    # importance from EARLIER tasks is retained too.
                    if client_fisher[cid] is None:
                        client_fisher[cid] = fisher
                        client_optpar[cid] = opt_par
                    else:
                        for k in fisher:
                            client_fisher[cid][k] = (
                                client_fisher[cid].get(
                                    k, torch.zeros_like(fisher[k]))
                                + fisher[k]
                            )
                        client_optpar[cid] = opt_par

            # ---- Snapshot: accuracy on TASK t's own held-out test
            # set, measured right now (A_{t,t}) ----
            task_test_loader = test_loaders_by_task[task]
            m = evaluate_model(global_model, task_test_loader)
            per_task_acc_when_learned[task] = (
                m["accuracy"] * 100 if m is not None else None
            )
            print(f"  >> Snapshot acc on Task {task+1} test data "
                  f"right after learning it: "
                  f"{per_task_acc_when_learned[task]}")

    # -- Final evaluation on each task's held-out test set, AFTER
    #    all tasks are done (A_{T,t}) --------------------------------
    per_task_acc_final = []
    for t in range(NUM_TASKS):
        m = evaluate_model(global_model, test_loaders_by_task[t])
        per_task_acc_final.append(m["accuracy"] * 100 if m is not None else None)

    bwt = compute_bwt(per_task_acc_when_learned, per_task_acc_final)

    # -- Final evaluation on the FULL held-out test set ---------------
    print(f"\n  Evaluating on held-out test patients (all classes)...")
    metrics = evaluate_model(global_model, test_loader_full)

    print(f"\n  FINAL TEST RESULTS -- {name} (seed={seed})")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1  : {metrics['f1']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  BWT       : {bwt if bwt is None else f'{bwt:.4f}'}")
    print(f"  Avg Comm  : {np.mean(round_comm_cost):,.0f} params/round")

    if "FedAvg + EWC + LoRA" in name:
        print(f"\n  Per-class report (TEST SET):")
        print(classification_report(
            metrics['true'], metrics['preds'],
            target_names=CLASS_NAMES, zero_division=0))
        print(f"\n  Confusion Matrix:")
        print(metrics['cm'])

    return {
        'name'        : name,
        'seed'        : seed,
        'test_acc'    : round(metrics['accuracy'] * 100, 2),
        'test_f1'     : round(metrics['f1'] * 100, 2),
        'test_prec'   : round(metrics['precision'] * 100, 2),
        'test_rec'    : round(metrics['recall'] * 100, 2),
        'bwt'         : round(bwt, 4) if bwt is not None else None,
        'avg_comm'    : int(np.mean(round_comm_cost)),
        'val_accs'    : round_val_accs,
        'per_task_acc_when_learned': per_task_acc_when_learned,
        'per_task_acc_final'      : per_task_acc_final,
        'trainable'   : global_model.trainable_params(),
        'total'       : global_model.total_params()
    }


# ============================================================
# PART 9 -- AGGREGATION ACROSS SEEDS
# ============================================================
def summarize_across_seeds(all_runs):
    """
    all_runs: list of result-dicts (one per (experiment, seed) pair).
    Returns: dict keyed by experiment name -> {metric: (mean, std)}
    """
    by_name = {}
    for r in all_runs:
        by_name.setdefault(r['name'], []).append(r)

    summary = {}
    metrics_to_agg = ['test_acc', 'test_f1', 'test_prec',
                       'test_rec', 'bwt', 'avg_comm']
    for name, runs in by_name.items():
        summary[name] = {}
        for metric in metrics_to_agg:
            vals = [r[metric] for r in runs if r[metric] is not None]
            if vals:
                summary[name][metric] = (
                    float(np.mean(vals)), float(np.std(vals)))
            else:
                summary[name][metric] = (None, None)
        summary[name]['n_seeds'] = len(runs)
    return summary


# ============================================================
# PART 10 -- MAIN
# ============================================================
if __name__ == "__main__":

    SPLIT_SEED = SEEDS[0]   # fixed train/val split across seeds,
                              # so only model init / training
                              # stochasticity varies across seeds
    set_seed(SPLIT_SEED)

    # -- Load data once (shared across all seeds/experiments;
    #    only the model training is reseeded per run) --------------
    (client_loaders, test_loader_full, test_loaders_by_task,
     class_weights) = create_federated_clients(SPLIT_SEED)

    experiments = [
        {"name": "1. FedAvg",                  "fedavg": True,  "ewc": False, "lora": False},
        {"name": "2. LoRA (Local Only)",        "fedavg": False, "ewc": False, "lora": True},
        {"name": "3. EWC (Local Only)",         "fedavg": False, "ewc": True,  "lora": False},
        {"name": "4. FedAvg + EWC",             "fedavg": True,  "ewc": True,  "lora": False},
        {"name": "5. FedAvg + LoRA",            "fedavg": True,  "ewc": False, "lora": True},
        {"name": "6. EWC + LoRA (Local Only)",  "fedavg": False, "ewc": True,  "lora": True},
        {"name": "7. FedAvg + EWC + LoRA",      "fedavg": True,  "ewc": True,  "lora": True},
    ]

    all_runs = []
    for seed in SEEDS:
        print(f"\n\n{'#'*72}\n  SEED {seed}\n{'#'*72}")
        set_seed(seed)
        for exp in experiments:
            r = run_experiment(
                exp['name'], exp['fedavg'], exp['ewc'], exp['lora'],
                client_loaders, test_loader_full, test_loaders_by_task,
                class_weights, seed
            )
            all_runs.append(r)

    # -- Save raw per-seed results --------------------------------
    raw_path = os.path.join(SCRIPT_DIR, "pefcl_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nSaved raw per-seed results to {raw_path}")

    # -- Aggregate mean +/- std across seeds -----------------------
    summary = summarize_across_seeds(all_runs)

    print("\n\n" + "="*88)
    print(f"  FINAL RESULTS (mean +/- std over {len(SEEDS)} seeds)")
    print("  MIT-BIH Arrhythmia | Inter-Patient Split | Class-Incremental CL")
    print("="*88)
    header = (f"  {'Method':<28} {'Acc':>14} {'F1':>14} "
              f"{'Prec':>14} {'Rec':>14} {'BWT':>14}")
    print(header)
    print(f"  {'-'*86}")

    def fmt(mean, std, scale=1.0):
        if mean is None:
            return "n/a"
        return f"{mean*scale:6.2f}+/-{std*scale:4.2f}"

    base_comm_mean = summary["1. FedAvg"]["avg_comm"][0]

    for exp in experiments:
        name = exp['name']
        s = summary[name]
        print(f"  {name:<28} "
              f"{fmt(*s['test_acc']):>14} "
              f"{fmt(*s['test_f1']):>14} "
              f"{fmt(*s['test_prec']):>14} "
              f"{fmt(*s['test_rec']):>14} "
              f"{fmt(*s['bwt']):>14}")

    print("="*88)
    print(f"\n  Communication cost (mean params/round, +/- std over seeds):")
    for exp in experiments:
        s = summary[exp['name']]
        mean_c, std_c = s['avg_comm']
        comm_pct = (f"(down {(1 - mean_c/base_comm_mean)*100:.0f}% vs FedAvg)"
                    if mean_c is not None and mean_c < base_comm_mean
                    else "(baseline)" if mean_c == base_comm_mean else "")
        print(f"  {exp['name']:<28} {mean_c:>10,.0f} +/- {std_c:>8,.0f} {comm_pct}")
    print("="*88)

    # -- Key findings (using mean values) --------------------------
    s1 = summary["1. FedAvg"]
    s5 = summary["5. FedAvg + LoRA"]
    s7 = summary["7. FedAvg + EWC + LoRA"]

    comm_saving = (1 - s5['avg_comm'][0] / s1['avg_comm'][0]) * 100
    f1_gain     = s7['test_f1'][0] - s5['test_f1'][0]
    bwt_gain    = (s7['bwt'][0] - s5['bwt'][0]
                   if s7['bwt'][0] is not None and s5['bwt'][0] is not None
                   else None)

    print(f"\n  KEY FINDINGS (mean over {len(SEEDS)} seeds):")
    print(f"  1. LoRA reduces communication by {comm_saving:.0f}% "
          f"({s1['avg_comm'][0]:,.0f} -> {s5['avg_comm'][0]:,.0f} params/round)")
    print(f"  2. Adding EWC to FedAvg+LoRA changes F1 by "
          f"{f1_gain:+.2f} points")
    if bwt_gain is not None:
        print(f"  3. Adding EWC changes BWT by {bwt_gain:+.4f} "
              f"(higher/less negative = less forgetting)")
    print(f"  4. PE-FCL (method 7) achieves "
          f"{s7['test_acc'][0]:.2f}% +/- {s7['test_acc'][1]:.2f}% acc, "
          f"{s7['test_f1'][0]:.2f}% +/- {s7['test_f1'][1]:.2f}% F1 "
          f"over {len(SEEDS)} seeds, with {comm_saving:.0f}% less "
          f"communication than FedAvg alone.")
    print("="*88)