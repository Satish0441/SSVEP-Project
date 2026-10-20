import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure imports work when launched from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn_ssvep_v2 import SSVEPGNNv2, normalize_adjacency


# ============================================================
# Configuration
# ============================================================

DATA_DIR = PROJECT_ROOT / "paper_experiment" / "dataset" / "1.0s"
GRAPH_FILE = PROJECT_ROOT / "paper_experiment" / "graph_9ch.npz"
OUTPUT_DIR = PROJECT_ROOT / "paper_experiment" / "loso_gnn_v2_results"

NUM_SUBJECTS = 35
NUM_CLASSES = 40
BLOCKS_PER_CLASS = 6

BATCH_SIZE = 128
LEARNING_RATE = 0.0002
EPOCHS = 100
SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Data loading
# ============================================================

def load_subject(subject_id):
    path = DATA_DIR / f"S{subject_id}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Missing subject file: {path}")

    with np.load(path) as data:
        X = data["X"]

    # Stored shape: [channels, samples, targets, blocks]
    if X.ndim != 4:
        raise ValueError(f"S{subject_id}: unexpected X shape {X.shape}")

    if X.shape[0] != 9 or X.shape[2:] != (
        NUM_CLASSES,
        BLOCKS_PER_CLASS,
    ):
        raise ValueError(f"S{subject_id}: unexpected X shape {X.shape}")

    # Convert to [trials, channels, samples].
    X = np.transpose(X, (2, 3, 0, 1))
    X = X.reshape(
        NUM_CLASSES * BLOCKS_PER_CLASS,
        X.shape[2],
        X.shape[3],
    )

    # Six blocks for each target.
    y = np.repeat(
        np.arange(NUM_CLASSES),
        BLOCKS_PER_CLASS,
    )

    return X.astype(np.float32), y.astype(np.int64)


# ============================================================
# Graph
# ============================================================

def load_graph():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Missing graph file: {GRAPH_FILE}")

    with np.load(GRAPH_FILE) as graph:
        adjacency = graph["adjacency"].astype(np.float32)

    if adjacency.shape != (9, 9):
        raise ValueError(
            f"Expected a 9x9 adjacency matrix, got {adjacency.shape}"
        )

    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("Adjacency matrix is not symmetric.")

    adjacency = torch.tensor(adjacency, dtype=torch.float32)
    return normalize_adjacency(adjacency).to(DEVICE)


# ============================================================
# One LOSO fold
# ============================================================

def run_fold(target_subject, adj_norm):
    set_seed(SEED)

    train_X_parts = []
    train_y_parts = []

    # Load all subjects except the held-out target.
    for subject_id in range(1, NUM_SUBJECTS + 1):
        if subject_id == target_subject:
            continue

        X, y = load_subject(subject_id)
        train_X_parts.append(X)
        train_y_parts.append(y)

    train_X = np.concatenate(train_X_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)

    train_dataset = TensorDataset(
        torch.from_numpy(train_X),
        torch.from_numpy(train_y),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model = SSVEPGNNv2(
        num_channels=9,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Train for the predetermined number of epochs.
    # The held-out subject is not loaded or evaluated here.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X_batch, adj_norm)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == y_batch
            ).sum().item()
            total_samples += batch_size

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"S{target_subject:02d} | "
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Loss {total_loss / total_samples:.4f} | "
                f"Train accuracy "
                f"{100 * total_correct / total_samples:.2f}%",
                flush=True,
            )

    # Only after training is complete, load and evaluate the target.
    test_X, test_y = load_subject(target_subject)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        test_tensor = torch.from_numpy(test_X)

        for start in range(0, len(test_X), BATCH_SIZE):
            X_batch = test_tensor[start:start + BATCH_SIZE].to(DEVICE)
            y_batch = torch.from_numpy(
                test_y[start:start + BATCH_SIZE]
            ).to(DEVICE)

            logits = model(X_batch, adj_norm)
            correct += (
                logits.argmax(dim=1) == y_batch
            ).sum().item()
            total += y_batch.size(0)

    accuracy = 100.0 * correct / total

    print(
        f"Completed target S{target_subject:02d}: "
        f"{accuracy:.2f}% ({correct}/{total})",
        flush=True,
    )

    return {
        "subject": target_subject,
        "accuracy_percent": accuracy,
        "correct_trials": correct,
        "total_trials": total,
        "epochs": EPOCHS,
        "seed": SEED,
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_subjects = [
        DATA_DIR / f"S{i}.npz"
        for i in range(1, NUM_SUBJECTS + 1)
    ]

    missing = [str(path) for path in expected_subjects if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing subject files:\n" + "\n".join(missing)
        )

    print("GNN v2 full LOSO experiment")
    print(f"Dataset window : {DATA_DIR.name}")
    print(f"Subjects       : {NUM_SUBJECTS}")
    print(f"Epochs/fold    : {EPOCHS}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Device         : {DEVICE}")
    print("Target labels are not used during training or model selection.")
    print()

    adj_norm = load_graph()
    results = []

    results_path = OUTPUT_DIR / "gnn_v2_loso_1.0s.csv"

    for target_subject in range(1, NUM_SUBJECTS + 1):
        print(f"\n{'=' * 60}")
        print(
            f"Fold {target_subject}/{NUM_SUBJECTS}: "
            f"held-out subject S{target_subject}"
        )
        print("=" * 60)

        result = run_fold(target_subject, adj_norm)
        results.append(result)

        # Save progress after every completed fold.
        with results_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

        accuracies = [r["accuracy_percent"] for r in results]
        print(
            f"Completed folds: {len(results)}/{NUM_SUBJECTS} | "
            f"Current mean: {np.mean(accuracies):.2f}%"
        )

    accuracies = np.array(
        [r["accuracy_percent"] for r in results],
        dtype=np.float64,
    )

    print("\n" + "=" * 60)
    print("FINAL LOSO RESULTS")
    print("=" * 60)
    print(f"Mean accuracy: {accuracies.mean():.2f}%")
    print(f"Std accuracy : {accuracies.std(ddof=1):.2f}%")
    print(f"Results CSV  : {results_path}")


if __name__ == "__main__":
    main()
