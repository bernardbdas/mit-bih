"""
Federated learning client-side training and adapter wrappers.

This module provides basic non-continual training update functions for local clients
as well as a client adapter wrapper for integrating with the Flower (flwr) simulation framework.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mit_bih.cl.trainer import FocalLoss, train_local_continual

# Optional integration with Flower
try:
    import flwr
    import flwr.client
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False


def train_local_client(
    model: nn.Module,
    dataloader: DataLoader | None,
    epochs: int = 1,
    lr: float = 0.001,
    device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """
    Performs standard (non-continual, non-EWC) local training updates on client data.

    Used as a baseline comparison.

    Args:
        model: The model to train.
        dataloader: DataLoader containing the client's training partition.
        epochs: Number of local training epochs.
        lr: Learning rate for optimizer.
        device: The computing device.

    Returns:
        The updated model state dictionary.
    """
    if dataloader is None or len(dataloader) == 0:
        return model.state_dict()

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def get_weights(model: nn.Module, use_lora: bool, device: torch.device | str = "cpu") -> list[np.ndarray]:
    """
    Serializes model parameters into a list of NumPy arrays for federated communication.

    For LoRA models, only extracts adapter parameters. For full parameters, extracts
    the entire model's state dictionary parameters in sorted key order.

    Args:
        model: PyTorch model.
        use_lora: If True, extracts only the adapter weights ('A' and 'B').
        device: The model's computing device.

    Returns:
        A list of numpy ndarrays.
    """
    if use_lora:
        params = model.get_lora_params()
        return [params['A'].cpu().numpy(), params['B'].cpu().numpy()]
    else:
        state_dict = model.state_dict()
        keys = sorted(state_dict.keys())
        return [state_dict[k].cpu().numpy() for k in keys]


def set_weights(model: nn.Module, use_lora: bool, ndarrays: list[np.ndarray], device: torch.device | str = "cpu") -> None:
    """
    Deserializes and updates model parameters from a list of NumPy arrays.

    Args:
        model: PyTorch model.
        use_lora: If True, updates only the adapter weights.
        ndarrays: A list of numpy arrays representing updated parameters.
        device: The model's computing device.
    """
    if use_lora:
        params = {
            'A': torch.from_numpy(ndarrays[0]).to(device),
            'B': torch.from_numpy(ndarrays[1]).to(device)
        }
        model.set_lora_params(params)
    else:
        state_dict = model.state_dict()
        keys = sorted(state_dict.keys())
        new_state_dict = {}
        for k, arr in zip(keys, ndarrays):
            new_state_dict[k] = torch.from_numpy(arr).to(device)
        model.load_state_dict(new_state_dict, strict=True)


# Schema to cache EWC state across stateless Flower simulation rounds
# Structure: { client_id: {"fisher": fisher_dict, "opt_par": opt_par_dict} }
client_ewc_states: dict[str, dict] = {}


if HAS_FLWR:
    class ArrhythmiaFlowerClient(flwr.client.NumPyClient):
        """
        Flower client wrapper implementing the NumPyClient interface.

        Facilitates local training and evaluation inside a simulated Flower server.
        """
        def __init__(
            self,
            cid: str,
            client_loaders_list: list[dict],
            test_loader_full: DataLoader,
            test_loaders_by_task: list[DataLoader | None],
            class_weights: torch.Tensor,
            use_ewc: bool,
            use_lora: bool,
            pretrained_backbone: nn.Module | None = None,
            device: torch.device | str = "cpu",
            dropout_rate: float = 0.3,
            lora_rank: int = 8,
            lora_alpha: float = 1.0,
            ewc_lambda_full: float = 50.0,
            ewc_lambda_lora: float = 1.0,
            rounds_per_task: int = 2,
            patience: int = 3
        ):
            from mit_bih.models.cnn import ECGCNN

            self.cid = str(cid)
            self.client_loader = client_loaders_list[int(cid)]
            self.test_loader_full = test_loader_full
            self.test_loaders_by_task = test_loaders_by_task
            self.class_weights = class_weights
            self.use_ewc = use_ewc
            self.use_lora = use_lora
            self.device = torch.device(device)
            self.dropout_rate = dropout_rate
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.ewc_lambda_full = ewc_lambda_full
            self.ewc_lambda_lora = ewc_lambda_lora
            self.rounds_per_task = rounds_per_task
            self.patience = patience

            # Initialize local client model
            self.model = ECGCNN(
                use_lora=use_lora,
                dropout_rate=dropout_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            ).to(self.device)

            # Load the pre-trained feature backbone if applicable
            if use_lora and pretrained_backbone is not None:
                self.model.load_pretrained_backbone(pretrained_backbone)

            self.criterion = FocalLoss(gamma=2.0)

        def get_parameters(self, config: dict) -> list[np.ndarray]:
            return get_weights(self.model, self.use_lora, self.device)

        def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
            from mit_bih.cl.ewc import compute_fisher

            # Sync parameters with the latest global updates
            set_weights(self.model, self.use_lora, parameters, self.device)

            task_num = int(config.get("task_num", 0))
            local_epochs = int(config.get("local_epochs", 1))
            lr = float(config.get("lr", 0.001))
            server_round = int(config.get("server_round", 1))

            train_loader = self.client_loader['train_by_task'][task_num]
            val_loader = self.client_loader['val_by_task'][task_num]

            if train_loader is None or len(train_loader) == 0:
                print(f"      [Client {self.cid}] No training data for Task {task_num+1}. Skipping local fit.")
                return self.get_parameters(config={}), 0, {}

            # Load cached client EWC histories
            fisher = None
            opt_par = None
            if self.use_ewc and self.cid in client_ewc_states:
                fisher = client_ewc_states[self.cid].get("fisher")
                opt_par = client_ewc_states[self.cid].get("opt_par")

            # Execute continual training local updates
            self.model = train_local_continual(
                self.model, train_loader, val_loader, self.criterion,
                local_epochs=local_epochs, lr=lr, patience=self.patience,
                use_ewc=self.use_ewc, fisher=fisher, opt_par=opt_par,
                ewc_lambda_full=self.ewc_lambda_full, ewc_lambda_lora=self.ewc_lambda_lora,
                device=self.device
            )

            # Compute and cache EWC Fisher matrix if this is the final round of the task
            is_last_round_of_task = (server_round % self.rounds_per_task == 0)
            if self.use_ewc and is_last_round_of_task:
                print(f"      [Client {self.cid}] Task {task_num+1} ending. Computing EWC Fisher matrices...")
                f_dict, o_dict = compute_fisher(self.model, train_loader, self.criterion, device=self.device)
                
                if self.cid not in client_ewc_states:
                    client_ewc_states[self.cid] = {"fisher": f_dict, "opt_par": o_dict}
                else:
                    for k in f_dict:
                        client_ewc_states[self.cid]["fisher"][k] = (
                            client_ewc_states[self.cid]["fisher"].get(k, torch.zeros_like(f_dict[k])) + f_dict[k]
                        )
                    client_ewc_states[self.cid]["opt_par"] = o_dict

            num_examples = len(train_loader.dataset)
            return self.get_parameters(config={}), num_examples, {}

        def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
            # Sync parameters with the latest global updates
            set_weights(self.model, self.use_lora, parameters, self.device)

            task_num = int(config.get("task_num", 0))
            test_loader = self.test_loaders_by_task[task_num]

            if test_loader is None or len(test_loader) == 0:
                print(f"      [Client {self.cid}] No testing data for Task {task_num+1}. Skipping local evaluation.")
                return 0.0, 0, {"accuracy": 0.0}

            self.model.eval()
            correct, total, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss_sum += loss.item() * len(labels)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            loss = loss_sum / total if total > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            return float(loss), int(total), {"accuracy": float(accuracy)}
else:
    # Fallback placeholder to maintain structural imports without flwr installed
    class ArrhythmiaFlowerClient:  # type: ignore
        """
        Placeholder class. Please install flwr to use ArrhythmiaFlowerClient.
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Flower (flwr) is not installed. "
                "Please run with standard simulation or install flwr."
            )
