import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gnn_ssvep import SSVEPGNN, normalize_adjacency


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("paper_experiment/dataset/1.0s")
GRAPH_FILE = Path("paper_experiment/graph_9ch.npz")

TARGET_SUBJECT = 1

BATCH_SIZE = 128
LEARNING_RATE = 0.0002
EPOCHS = 100

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Load one subject
# ============================================================

def load_subject(subject_id):

    path = DATA_DIR / f"S{subject_id}.npz"

    data = np.load(path)

    X = data["X"]

    # X:
    # [channels, samples, targets, blocks]
    #
    # Convert to:
    # [targets * blocks, channels, samples]

    X = np.transpose(X, (2, 3, 0, 1))

    X = X.reshape(
        X.shape[0] * X.shape[1],
        X.shape[2],
        X.shape[3],
    )

    # Labels:
    # target 0 repeated for all 6 blocks,
    # target 1 repeated for all 6 blocks, ...

    y = np.repeat(
        np.arange(40),
        6,
    )

    return X.astype(np.float32), y.astype(np.int64)


# ============================================================
# Load training subjects
# ============================================================

train_X = []
train_y = []

for subject_id in range(1, 36):

    if subject_id == TARGET_SUBJECT:
        continue

    X, y = load_subject(subject_id)

    train_X.append(X)
    train_y.append(y)


train_X = np.concatenate(train_X, axis=0)
train_y = np.concatenate(train_y, axis=0)


# ============================================================
# Load target subject
# ============================================================

test_X, test_y = load_subject(TARGET_SUBJECT)


print("Training subjects : 34")
print("Target subject    :", TARGET_SUBJECT)

print("Train X shape     :", train_X.shape)
print("Train y shape     :", train_y.shape)

print("Test X shape      :", test_X.shape)
print("Test y shape      :", test_y.shape)


# ============================================================
# Convert to tensors
# ============================================================

train_X = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_y)

test_X = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_y)


train_dataset = TensorDataset(
    train_X,
    train_y,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)


# ============================================================
# Load actual graph
# ============================================================

graph = np.load(GRAPH_FILE)

adjacency = torch.tensor(
    graph["adjacency"],
    dtype=torch.float32,
)

adj_norm = normalize_adjacency(adjacency)


# ============================================================
# Model
# ============================================================

device = torch.device("cpu")

model = SSVEPGNN(
    num_channels=9,
    num_classes=40,
).to(device)

adj_norm = adj_norm.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training
# ============================================================

print()
print("Starting training...")
print()

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(
            X_batch,
            adj_norm,
        )

        loss = criterion(
            logits,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X_batch.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)


    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total


    # --------------------------------------------------------
    # Target-subject evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            test_X.to(device),
            adj_norm,
        )

        predictions = logits.argmax(dim=1)

        test_accuracy = (
            predictions == test_y.to(device)
        ).float().mean().item() * 100.0


    print(
        f"Epoch {epoch:02d} | "
        f"Loss {train_loss:.4f} | "
        f"Train {train_accuracy:.2f}% | "
        f"S{TARGET_SUBJECT} {test_accuracy:.2f}%"
    )


print()
print("Finished.")
