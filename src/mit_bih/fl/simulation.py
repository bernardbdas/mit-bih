"""
Federated Continual Learning simulation runners.

This module provides two simulation orchestrators:
1. `run_local_simulation`: Performs in-memory federated learning simulation across clients
   using standard PyTorch training and EWC/LoRA constraints.
2. `run_flower_simulation`: Runs a simulation using the Flower (flwr) simulation framework.
"""

import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mit_bih.models.cnn import ECGCNN
from mit_bih.cl.trainer import FocalLoss, train_local_continual, evaluate_model
from mit_bih.cl.ewc import compute_fisher
from mit_bih.fl.server import federated_average_dict
from mit_bih.fl.client import get_weights, set_weights, HAS_FLWR

if HAS_FLWR:
    import flwr
    from mit_bih.fl.client import ArrhythmiaFlowerClient, client_ewc_states


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across packages."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_backbone(
    client_loaders: list[dict],
    pretrain_rounds: int = 1,
    local_epochs: int = 1,
    lr: float = 0.001,
    patience: int = 3,
    dropout_rate: float = 0.3,
    use_fedavg: bool = True,
    device: torch.device | str = "cpu"
) -> nn.Module:
    """
    Pre-trains the feature extractor backbone on Task 1 data.

    Averages client updates using FedAvg to build a robust general representation
    before starting continual classification.

    Args:
        client_loaders: List of client data loader dictionaries.
        pretrain_rounds: Number of federated pre-training communication rounds.
        local_epochs: Epochs trained locally per client per round.
        lr: Learning rate.
        patience: Validation early stopping patience.
        dropout_rate: Dropout rate for CNN.
        use_fedavg: If True, aggregates weights. If False, copies client 0 weights.
        device: The computing device.

    Returns:
        The pre-trained ECGCNN model.
    """
    print(f"\n  {'-'*51}")
    print(f"  Pretraining backbone on Task 1 data ({pretrain_rounds} rounds, fedavg={use_fedavg})")
    print(f"  {'-'*51}")

    pretrain_model = ECGCNN(use_lora=False, dropout_rate=dropout_rate).to(device)
    num_clients = len(client_loaders)
    client_models = [copy.deepcopy(pretrain_model) for _ in range(num_clients)]
    client_sizes_task1 = [cl['task_sizes'][0] for cl in client_loaders]
    criterion = FocalLoss(gamma=2.0)

    for rnd in range(pretrain_rounds):
        local_weights = []
        for cid in range(num_clients):
            train_loader = client_loaders[cid]['train_by_task'][0]
            val_loader = client_loaders[cid]['val_by_task'][0]
            client_models[cid] = train_local_continual(
                client_models[cid], train_loader, val_loader, criterion,
                local_epochs=local_epochs, lr=lr, patience=patience,
                use_ewc=False, device=device
            )
            local_weights.append(copy.deepcopy(client_models[cid].state_dict()))

        if use_fedavg:
            agg = federated_average_dict(local_weights, client_sizes_task1)
            pretrain_model.load_state_dict(agg)
            for cid in range(num_clients):
                client_models[cid].load_state_dict(copy.deepcopy(agg))
        else:
            pretrain_model = copy.deepcopy(client_models[0])

        # Validate on client 0 val loader
        val_loader = client_loaders[0]['val_by_task'][0]
        if val_loader is not None and len(val_loader) > 0:
            pretrain_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = pretrain_model(inputs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = 100.0 * correct / total if total > 0 else 0.0
            print(f"    Pretrain round {rnd+1}/{pretrain_rounds} | client0 val acc: {val_acc:.2f}%")

    print("  Backbone pretraining done.\n")
    return pretrain_model


def run_local_simulation(
    name: str,
    use_fedavg: bool,
    use_ewc: bool,
    use_lora: bool,
    client_loaders: list[dict],
    test_loader_full: DataLoader,
    test_loaders_by_task: list[DataLoader | None],
    class_weights: torch.Tensor,
    seed: int = 42,
    num_rounds: int = 2,
    rounds_per_task: int = 1,
    local_epochs: int = 1,
    lr: float = 0.001,
    patience: int = 5,
    dropout_rate: float = 0.3,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    ewc_lambda_full: float = 50.0,
    ewc_lambda_lora: float = 1.0,
    pretrain_rounds: int = 1,
    device: torch.device | str = "cpu"
) -> dict:
    """
    Orchestrates an in-memory Federated Continual Learning simulation.

    Args:
        name: Name of the experiment.
        use_fedavg: If True, aggregates parameters using FedAvg after each round.
        use_ewc: If True, computes and regularizes with EWC penalties.
        use_lora: If True, trains only LoRA adapter layers.
        client_loaders: Partitioned client data loaders.
        test_loader_full: Full held-out test loader.
        test_loaders_by_task: Test loaders separated by continual task classes.
        class_weights: Imbalance class weights.
        seed: Random seed.
        num_rounds: Total communication rounds.
        rounds_per_task: Rounds to train per task.
        local_epochs: Local epochs per round.
        lr: Learning rate.
        patience: Early stopping patience.
        dropout_rate: Dropout rate.
        lora_rank: LoRA adapter rank.
        lora_alpha: LoRA scaling factor.
        ewc_lambda_full: EWC scaling multiplier (Full).
        ewc_lambda_lora: EWC scaling multiplier (LoRA).
        pretrain_rounds: Backbone pre-training rounds.
        device: Computing target.

    Returns:
        A dictionary containing run metrics (accuracy, F1, precision, recall, comm cost).
    """
    print(f"\n{'='*55}")
    print(f"  Experiment: {name}  (seed={seed})")
    print(f"  fedavg={use_fedavg} | ewc={use_ewc} | lora={use_lora}")
    print(f"{'='*55}")

    set_seed(seed)
    num_clients = len(client_loaders)
    criterion = FocalLoss(gamma=2.0)

    # Initialize model
    global_model = ECGCNN(
        use_lora=use_lora,
        dropout_rate=dropout_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    ).to(device)

    # Pre-train the feature extractor if LoRA adapter is used
    if use_lora:
        pretrained = pretrain_backbone(
            client_loaders,
            pretrain_rounds=pretrain_rounds,
            local_epochs=local_epochs,
            lr=lr,
            patience=patience,
            dropout_rate=dropout_rate,
            use_fedavg=use_fedavg,
            device=device
        )
        global_model.load_pretrained_backbone(pretrained)

    print(f"  Trainable params: {global_model.trainable_params():,} / {global_model.total_params():,} "
          f"({global_model.trainable_params()/global_model.total_params()*100:.1f}%)")

    # Client EWC memory registers
    client_fisher = [None] * num_clients
    client_optpar = [None] * num_clients

    round_val_accs = []
    round_comm_cost = []
    num_tasks = num_rounds // rounds_per_task
    per_task_acc_when_learned = [None] * num_tasks

    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    for rnd in range(num_rounds):
        task = rnd // rounds_per_task
        is_last_round_of_task = (rnd + 1) % rounds_per_task == 0
        print(f"\n  Round {rnd+1}/{num_rounds} | Task {task+1}/{num_tasks}")

        local_weights = []
        round_comm = 0
        round_val_sum = 0.0
        round_val_n = 0

        client_sizes_this_task = [cl['task_sizes'][task] for cl in client_loaders]
        print(f"  Client sizes for Task {task+1}: {client_sizes_this_task}")

        for cid in range(num_clients):
            print(f"\n    Client {cid+1}")
            train_loader = client_loaders[cid]['train_by_task'][task]
            val_loader = client_loaders[cid]['val_by_task'][task]

            client_models[cid] = train_local_continual(
                client_models[cid], train_loader, val_loader, criterion,
                local_epochs=local_epochs, lr=lr, patience=patience,
                use_ewc=use_ewc, fisher=client_fisher[cid], opt_par=client_optpar[cid],
                ewc_lambda_full=ewc_lambda_full, ewc_lambda_lora=ewc_lambda_lora,
                device=device
            )

            # Extract weights to transmit
            if use_lora:
                lp = client_models[cid].get_lora_params()
                w = {'A': lp['A'], 'B': lp['B']}
            else:
                w = copy.deepcopy(client_models[cid].state_dict())

            local_weights.append(w)
            # Count parameters transmitted (communication cost)
            round_comm += sum(v.numel() for v in w.values())

            # Evaluate local client validation accuracy
            if val_loader is not None and len(val_loader) > 0:
                client_models[cid].eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = client_models[cid](inputs)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                val_acc = 100 * correct / total if total > 0 else 0.0
                round_val_sum += val_acc
                round_val_n += 1

        # Parameter aggregation
        if use_fedavg:
            agg = federated_average_dict(local_weights, client_sizes_this_task)
            if use_lora:
                for cid in range(num_clients):
                    client_models[cid].set_lora_params({'A': agg['A'], 'B': agg['B']})
                global_model.set_lora_params({'A': agg['A'], 'B': agg['B']})
            else:
                global_model.load_state_dict(agg)
                for cid in range(num_clients):
                    client_models[cid].load_state_dict(copy.deepcopy(agg))
        else:
            global_model = copy.deepcopy(client_models[0])

        avg_val = (round_val_sum / round_val_n) if round_val_n else 0.0
        round_val_accs.append(avg_val)
        round_comm_cost.append(round_comm)
        print(f"\n  Round {rnd+1} avg val acc: {avg_val:.2f}% | Comm params: {round_comm:,}")

        # Compute EWC Fisher memory registers at the end of task training
        if is_last_round_of_task:
            if use_ewc:
                for cid in range(num_clients):
                    fisher, opt_par = compute_fisher(
                        client_models[cid],
                        client_loaders[cid]['train_by_task'][task],
                        criterion, device=device
                    )
                    if client_fisher[cid] is None:
                        client_fisher[cid] = fisher
                        client_optpar[cid] = opt_par
                    else:
                        for k in fisher:
                            client_fisher[cid][k] = (
                                client_fisher[cid].get(k, torch.zeros_like(fisher[k])) + fisher[k]
                            )
                        client_optpar[cid] = opt_par

            # Record task snapshot accuracy immediately after learning
            task_test_loader = test_loaders_by_task[task]
            m = evaluate_model(global_model, task_test_loader, device=device)
            per_task_acc_when_learned[task] = (m["accuracy"] * 100 if m is not None else None)
            print(f"  >> Snapshot acc on Task {task+1} test data right after learning it: {per_task_acc_when_learned[task]}")

    # Final evaluation per individual task
    per_task_acc_final = []
    for t in range(num_tasks):
        m = evaluate_model(global_model, test_loaders_by_task[t], device=device)
        per_task_acc_final.append(m["accuracy"] * 100 if m is not None else None)

    # Compute Backward Transfer (BWT)
    diffs = []
    for t in range(num_tasks - 1):
        a_tt = per_task_acc_when_learned[t]
        a_Tt = per_task_acc_final[t]
        if a_tt is not None and a_Tt is not None:
            diffs.append(a_Tt - a_tt)
    bwt = float(np.mean(diffs)) if diffs else None

    # Evaluate on full held-out test dataset
    print(f"\n  Evaluating on held-out test patients (all classes)...")
    metrics = evaluate_model(global_model, test_loader_full, device=device)

    if metrics is None:
        metrics = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "preds": [], "true": []}

    print(f"\n  FINAL TEST RESULTS -- {name} (seed={seed})")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1  : {metrics['f1']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  BWT       : {bwt if bwt is None else f'{bwt:.4f}'}")
    print(f"  Avg Comm  : {np.mean(round_comm_cost):,.0f} params/round")

    return {
        'name': name,
        'seed': seed,
        'test_acc': round(metrics['accuracy'] * 100, 2),
        'test_f1': round(metrics['f1'] * 100, 2),
        'test_prec': round(metrics['precision'] * 100, 2),
        'test_rec': round(metrics['recall'] * 100, 2),
        'bwt': round(bwt, 4) if bwt is not None else None,
        'avg_comm': int(np.mean(round_comm_cost)),
        'val_accs': round_val_accs,
        'per_task_acc_when_learned': per_task_acc_when_learned,
        'per_task_acc_final': per_task_acc_final,
        'trainable': global_model.trainable_params(),
        'total': global_model.total_params()
    }


def run_flower_simulation(
    use_ewc: bool,
    use_lora: bool,
    name: str,
    client_loaders: list[dict],
    test_loader_full: DataLoader,
    test_loaders_by_task: list[DataLoader | None],
    class_weights: torch.Tensor,
    num_rounds: int = 4,
    rounds_per_task: int = 2,
    local_epochs: int = 1,
    lr: float = 0.001,
    patience: int = 3,
    dropout_rate: float = 0.3,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    ewc_lambda_full: float = 50.0,
    ewc_lambda_lora: float = 1.0,
    pretrain_rounds: int = 1,
    device: str = "cpu"
) -> None:
    """
    Runs a Federated Continual Learning simulation using the Flower (flwr) framework.

    Args:
        use_ewc: If True, uses EWC constraints.
        use_lora: If True, uses LoRA adapters.
        name: Experiment name.
        client_loaders: List of client loaders.
        test_loader_full: Test loader.
        test_loaders_by_task: Task-segmented test loaders.
        class_weights: Class weights.
        num_rounds: Communication rounds.
        rounds_per_task: Rounds per task.
        local_epochs: Epochs trained locally per round.
        lr: Learning rate.
        patience: Validation early stopping patience.
        dropout_rate: Dropout rate.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        ewc_lambda_full: EWC lambda (Full).
        ewc_lambda_lora: EWC lambda (LoRA).
        pretrain_rounds: Backbone pre-training rounds.
        device: CPU/GPU target.
    """
    if not HAS_FLWR:
        raise ImportError(
            "Flower (flwr) is not installed. "
            "Please install flwr to run run_flower_simulation."
        )

    print("\n" + "="*60)
    print(f"Starting Flower Simulation: {name}")
    print(f"use_ewc={use_ewc} | use_lora={use_lora}")
    print("="*60)

    # 1. Reset client EWC caches
    global client_ewc_states
    client_ewc_states.clear()

    # 2. Pre-train feature backbone if using LoRA
    pretrained_backbone = None
    if use_lora:
        pretrained_backbone = pretrain_backbone(
            client_loaders=client_loaders,
            pretrain_rounds=pretrain_rounds,
            local_epochs=local_epochs,
            lr=lr,
            patience=patience,
            dropout_rate=dropout_rate,
            use_fedavg=True,
            device=device
        )

    # 3. Instantiate a global reference model for server parameter initialization
    global_model = ECGCNN(
        use_lora=use_lora,
        dropout_rate=dropout_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    ).to(device)

    if use_lora and pretrained_backbone is not None:
        global_model.load_pretrained_backbone(pretrained_backbone)

    initial_params = flwr.common.ndarrays_to_parameters(
        get_weights(global_model, use_lora, device)
    )

    num_clients = len(client_loaders)

    # 4. Define client generator callable
    def client_fn(cid: str) -> flwr.client.Client:
        return ArrhythmiaFlowerClient(
            cid=cid,
            client_loaders_list=client_loaders,
            test_loader_full=test_loader_full,
            test_loaders_by_task=test_loaders_by_task,
            class_weights=class_weights,
            use_ewc=use_ewc,
            use_lora=use_lora,
            pretrained_backbone=pretrained_backbone,
            device=device,
            dropout_rate=dropout_rate,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            ewc_lambda_full=ewc_lambda_full,
            ewc_lambda_lora=ewc_lambda_lora,
            rounds_per_task=rounds_per_task,
            patience=patience
        ).to_client()

    # Configure helper routines passed to Flower server
    def fit_config(server_round: int):
        task_num = (server_round - 1) // rounds_per_task
        return {
            "task_num": task_num,
            "local_epochs": local_epochs,
            "lr": lr,
            "server_round": server_round
        }

    def evaluate_config(server_round: int):
        task_num = (server_round - 1) // rounds_per_task
        return {
            "task_num": task_num,
            "server_round": server_round
        }

    def evaluate_metrics_aggregation_fn(metrics):
        accuracies = [m["accuracy"] for _, m in metrics]
        examples = [num for num, _ in metrics]
        weighted_acc = sum(a * e for a, e in zip(accuracies, examples)) / sum(examples)
        return {"accuracy": weighted_acc}

    # 5. Configure FedAvg server strategy
    strategy = flwr.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_params
    )

    # 6. Start Flower Simulation
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )

    print(f"\nSimulation finished for {name}.")
    print(f"History of accuracy per round: {history.metrics_distributed}")
