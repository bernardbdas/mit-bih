import torch
import torch.nn as nn
import copy
import numpy as np
import wfdb
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.005
EWC_LAMBDA = 500.0

# MIT-BIH is highly imbalanced. These weights force the model to care about minority classes.
CLASS_WEIGHTS = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0]) 

# Data Path exactly as structured in your VS Code
DATA_PATH = 'mit-bih-arrhythmia-database-1.0.0/' 

# AAMI Standard Mapping for 5 Arrhythmia Classes
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # SVEB
    'V': 2, 'E': 2,                          # VEB
    'F': 3,                                  # Fusion
    '/': 4, 'f': 4, 'Q': 4                   # Unknown
}

# Standard inter-patient split to prevent data leakage 
TRAIN_PATIENTS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
TEST_PATIENTS = ['100', '103', '104', '105', '111', '113', '117', '121', '123', '200', '210', '212', '213', '214', '217', '219', '221', '222', '228', '231', '232', '233', '234']

# ==========================================
# 2. DATA PREPARATION WITH PREPROCESSING
# ==========================================
def extract_beats(patient_ids, window_size=90):
    """Extracts individual heartbeats centered around R-peaks and normalizes them."""
    X_all, y_all = [], []
    
    for pid in patient_ids:
        record_path = os.path.join(DATA_PATH, pid)
        if not os.path.exists(record_path + '.dat'):
            continue 
            
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        signal = record.p_signal[:, 0]
        peaks = annotation.sample
        symbols = annotation.symbol
        
        for peak, symbol in zip(peaks, symbols):
            if symbol in AAMI_MAPPING:
                if peak - window_size >= 0 and peak + window_size < len(signal):
                    beat = signal[peak - window_size : peak + window_size]
                    
                    # Z-SCORE NORMALIZATION 
                    mean_val = np.mean(beat)
                    std_val = np.std(beat)
                    if std_val > 0:
                        beat = (beat - mean_val) / std_val
                    else:
                        beat = beat - mean_val 
                        
                    X_all.append(beat)
                    y_all.append(AAMI_MAPPING[symbol])
                    
    X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32).unsqueeze(1) 
    y_tensor = torch.tensor(np.array(y_all), dtype=torch.long)
    return X_tensor, y_tensor

def get_iomt_dataloaders():
    print("Extracting & Preprocessing actual MIT-BIH ECG heartbeats (This may take a minute)...")
    
    X_test, y_test = extract_beats(TEST_PATIENTS)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    patients_per_client = len(TRAIN_PATIENTS) // NUM_CLIENTS
    client_loaders = []
    
    for i in range(NUM_CLIENTS):
        start_idx = i * patients_per_client
        end_idx = start_idx + patients_per_client if i != NUM_CLIENTS - 1 else len(TRAIN_PATIENTS)
        
        client_patients = TRAIN_PATIENTS[start_idx:end_idx]
        X_client, y_client = extract_beats(client_patients)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_client.numpy(), y_client.numpy(), test_size=0.2, random_state=42, stratify=y_client.numpy()
        )
        
        c_train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
        c_val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
        
        client_loaders.append({'train': c_train_loader, 'val': c_val_loader})
        print(f"Client {i+1} loaded {len(X_train)} train beats, {len(X_val)} val beats.")
        
    print(f"Global Test Set loaded: {len(X_test)} unseen beats.")
    return client_loaders, test_loader

# ==========================================
# 3. MODEL ARCHITECTURE (1D CNN + LoRA)
# ==========================================
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.W0 = nn.Linear(in_features, out_features, bias=False)
        self.W0.weight.requires_grad = False 
        
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.alpha = alpha
        
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.W0(x) + self.alpha * self.B(self.A(x))

class ECG_Model(nn.Module):
    def __init__(self, use_lora=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        linear_input = 32 * 45 
        
        if use_lora:
            self.classifier = LoRALayer(linear_input, 5)
        else:
            self.classifier = nn.Linear(linear_input, 5)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==========================================
# 4. EWC ALGORITHM COMPONENTS
# ==========================================
def compute_fisher(model, dataloader, criterion):
    fisher_dict = {}
    optpar_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
            optpar_dict[name] = param.data.clone()
            
    model.eval()
    for inputs, labels in dataloader:
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2 / len(dataloader)
                
    return fisher_dict, optpar_dict

def ewc_loss(model, fisher_dict, optpar_dict):
    loss = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_dict:
            _loss = fisher_dict[name] * (param - optpar_dict[name]) ** 2
            loss += _loss.sum()
    return loss * (EWC_LAMBDA / 2)

# ==========================================
# 5. FEDERATED TRAINING LOOP (WITH VALIDATION)
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiment(client_loaders, test_loader, use_fedavg, use_ewc, use_lora, exp_name):
    print(f"\n--- Starting Experiment: {exp_name} ---")
    
    global_model = ECG_Model(use_lora=use_lora)
    clients = [copy.deepcopy(global_model) for _ in range(NUM_CLIENTS)]
    client_fisher_histories = [{}] * NUM_CLIENTS 
    client_optpar_histories = [{}] * NUM_CLIENTS
    
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    
    for round_num in range(NUM_ROUNDS):
        print(f"  Round {round_num+1}/{NUM_ROUNDS}")
        local_weights = []
        comm_cost_tracker = 0
        
        for idx, client_model in enumerate(clients):
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, client_model.parameters()), lr=LEARNING_RATE)
            train_loader = client_loaders[idx]['train']
            val_loader = client_loaders[idx]['val']
            
            client_model.train()
            for epoch in range(LOCAL_EPOCHS):
                # Training pass
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if use_ewc and client_fisher_histories[idx]:
                        loss += ewc_loss(client_model, client_fisher_histories[idx], client_optpar_histories[idx])
                        
                    loss.backward()
                    optimizer.step()
                
                # Validation pass to check for overfitting
                client_model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        val_outputs = client_model(val_inputs)
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_predicted == val_labels).sum().item()
                
                val_acc = 100 * val_correct / val_total
                print(f"    [Client {idx+1}] Epoch {epoch+1} Local Val Accuracy: {val_acc:.2f}%")
                client_model.train()
            
            # EWC Setup for next round
            if use_ewc:
                f_dict, o_dict = compute_fisher(client_model, train_loader, criterion)
                client_fisher_histories[idx] = f_dict
                client_optpar_histories[idx] = o_dict
                
            # Weight Extraction
            if use_lora:
                w_to_send = {k: v for k, v in client_model.state_dict().items() if 'A' in k or 'B' in k}
            else:
                w_to_send = client_model.state_dict()
                
            local_weights.append(w_to_send)
            comm_cost_tracker += sum(p.numel() for p in w_to_send.values())
            
        # FedAvg Server Aggregation
        if use_fedavg:
            avg_weights = {}
            for key in local_weights[0].keys():
                avg_weights[key] = torch.mean(torch.stack([w[key] for w in local_weights]), dim=0)
            
            global_model.load_state_dict(avg_weights, strict=False)
            for client in clients:
                client.load_state_dict(global_model.state_dict(), strict=False)
        else:
            global_model = clients[0]

    # Global Unseen Patient Test
    test_acc = evaluate_model(global_model, test_loader)
    print(f"[{exp_name}] Final Global Test Accuracy: {test_acc:.2f}% | Parameters Communicated/Round: {comm_cost_tracker}")
    return test_acc

# ==========================================
# 6. RESEARCH EXECUTION 
# ==========================================
if __name__ == "__main__":
    c_loaders, t_loader = get_iomt_dataloaders()
    
    experiments = [
        {"name": "1. FedAvg",             "fedavg": True,  "ewc": False, "lora": False},
        {"name": "2. LoRA (Local Only)",  "fedavg": False, "ewc": False, "lora": True},
        {"name": "3. EWC (Local Only)",   "fedavg": False, "ewc": True,  "lora": False},
        {"name": "4. FedAvg + EWC",       "fedavg": True,  "ewc": True,  "lora": False},
        {"name": "5. FedAvg + LoRA",      "fedavg": True,  "ewc": False, "lora": True},
        {"name": "6. EWC + LoRA",         "fedavg": False, "ewc": True,  "lora": True},
        {"name": "7. FedAvg + EWC + LoRA","fedavg": True,  "ewc": True,  "lora": True}, 
    ]
    
    results = {}
    for exp in experiments:
        acc = run_experiment(c_loaders, t_loader, exp["fedavg"], exp["ewc"], exp["lora"], exp["name"])
        results[exp["name"]] = acc
        
    print("\n================ FINAL RESULTS ================")
    for name, acc in results.items():
        print(f"{name.ljust(25)}: {acc:.2f}% Accuracy")
    print("===============================================")