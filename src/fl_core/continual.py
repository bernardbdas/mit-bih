import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class FocalLoss(nn.Module):
    """
    Focal loss down-weights easy/well-classified examples and focuses gradient
    on hard/rare ones, adaptively, without a fixed per-class multiplier.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_fisher(model, dataloader, criterion, device):
    """
    Compute diagonal Fisher Information Matrix over TRAINABLE parameters only.
    """
    if dataloader is None:
        return {}, {}

    fisher = {}
    opt_par = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
            opt_par[name] = param.data.clone()

    model.eval()
    n_batches = len(dataloader)
    if n_batches == 0:
        return fisher, opt_par

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2) / n_batches

    return fisher, opt_par


def ewc_penalty(model, fisher, opt_par, ewc_lambda_full, ewc_lambda_lora, device):
    """
    EWC penalty added to the local loss while training on the CURRENT task.
    """
    if not fisher:
        return torch.tensor(0.0, device=device)

    ewc_lambda = ewc_lambda_lora if getattr(model, "use_lora", False) else ewc_lambda_full

    # Normalise Fisher so the penalty's overall scale stays bounded
    max_fisher = max(
        (f.max().item() for f in fisher.values() if f.numel() > 0),
        default=0.0
    )
    norm = max_fisher if max_fisher > 1e-12 else 1.0

    loss = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher:
            loss += (
                (fisher[name].to(device) / norm) *
                (param - opt_par[name].to(device)).pow(2)
            ).sum()
    return (ewc_lambda / 2) * loss


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_local_continual(
    model, train_loader, val_loader, criterion,
    local_epochs=10, lr=0.001, patience=5,
    use_ewc=False, fisher=None, opt_par=None,
    ewc_lambda_full=50.0, ewc_lambda_lora=1.0,
    device="cpu"
):
    """
    Train one client locally on the current task's data.
    """
    if train_loader is None or len(train_loader) == 0:
        return model

    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    early_stopper = EarlyStopping(patience=patience)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(local_epochs):
        if hasattr(model, "set_train_mode"):
            model.set_train_mode()
        else:
            model.train()

        running_loss = 0.0
        running_task_loss = 0.0
        running_ewc_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            task_loss = criterion(model(inputs), labels)
            loss = task_loss

            ewc_loss_val = torch.tensor(0.0, device=device)
            if use_ewc and fisher:
                ewc_loss_val = ewc_penalty(
                    model, fisher, opt_par,
                    ewc_lambda_full, ewc_lambda_lora, device
                )
                loss = loss + ewc_loss_val

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                1.0
            )
            optimizer.step()

            running_loss += loss.item()
            running_task_loss += task_loss.item()
            running_ewc_loss += ewc_loss_val.item()

        train_loss = running_loss / len(train_loader)
        avg_task_loss = running_task_loss / len(train_loader)
        avg_ewc_loss = running_ewc_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None and len(val_loader) > 0:
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for v_inputs, v_labels in val_loader:
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                    outputs = model(v_inputs)
                    total_loss += criterion(outputs, v_labels).item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == v_labels).sum().item()
                    total += v_labels.size(0)
            if total > 0:
                val_loss = total_loss / len(val_loader)
                val_acc = 100.0 * correct / total

        scheduler.step(val_loss)

        if use_ewc and avg_task_loss > 1e-8 and avg_ewc_loss > 5 * avg_task_loss:
            print(
                f"      WARNING: EWC penalty ({avg_ewc_loss:.4f}) >> task loss ({avg_task_loss:.4f}). "
                f"EWC may be dominating training."
            )

        print(
            f"      Epoch {epoch+1}/{local_epochs} | "
            f"Train Loss: {train_loss:.4f} (task={avg_task_loss:.4f} ewc={avg_ewc_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        early_stopper(val_loss)
        if early_stopper.stop:
            print("      Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    return model
