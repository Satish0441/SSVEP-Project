import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.gnn_ssvep_v2_shallow import SSVEPGNNv2, normalize_adjacency

DATA_DIR = Path("paper_experiment/dataset/1.0s")
GRAPH_FILE = Path("paper_experiment/graph_9ch.npz")

TARGET_SUBJECT = 1
VAL_SUBJECTS = [6, 7, 8, 9]
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
EPOCHS = 30
PATIENCE = 7
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_subject(subject_id):
    path = DATA_DIR / f"S{subject_id}.npz"
    with np.load(path) as data:
        X = data["X"]

    # [channels, samples, targets, blocks]
    X = np.transpose(X, (2, 3, 0, 1))
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])
    y = np.repeat(np.arange(40), 6)
    return X.astype(np.float32), y.astype(np.int64)

all_subjects = set(range(1, 36))
train_subjects = sorted(all_subjects - set(VAL_SUBJECTS) - {TARGET_SUBJECT})

assert len(train_subjects) == 30
assert not set(train_subjects) & set(VAL_SUBJECTS)
assert TARGET_SUBJECT not in train_subjects + VAL_SUBJECTS

def load_group(subject_ids):
    xs, ys = [], []
    for sid in subject_ids:
        X, y = load_subject(sid)
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)

train_X, train_y = load_group(train_subjects)
val_X, val_y = load_group(VAL_SUBJECTS)
test_X, test_y = load_subject(TARGET_SUBJECT)

print("Training subjects:", train_subjects)
print("Validation subjects:", VAL_SUBJECTS)
print("Held-out target:", TARGET_SUBJECT)
print("Train shape:", train_X.shape, train_y.shape)
print("Validation shape:", val_X.shape, val_y.shape)
print("Target shape:", test_X.shape, test_y.shape)

train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)

val_X = torch.from_numpy(val_X)
val_y = torch.from_numpy(val_y)
test_X = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_y)

graph = np.load(GRAPH_FILE)
adjacency = torch.tensor(graph["adjacency"], dtype=torch.float32)
adj_norm = normalize_adjacency(adjacency)

device = torch.device("cpu")
model = SSVEPGNNv2(num_channels=9, num_classes=40).to(device)
adj_norm = adj_norm.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_accuracy = -1.0
best_epoch = 0
best_state = None
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch, adj_norm)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = total_loss / total
    train_accuracy = 100.0 * correct / total

    model.eval()
    with torch.no_grad():
        val_logits = model(val_X.to(device), adj_norm)
        val_accuracy = (
            (val_logits.argmax(dim=1) == val_y.to(device))
            .float().mean().item() * 100.0
        )

    improved = val_accuracy > best_val_accuracy
    if improved:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:02d} | Loss {train_loss:.4f} | "
        f"Train {train_accuracy:.2f}% | "
        f"Validation {val_accuracy:.2f}%"
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

# Save the best checkpoint selected using source-subject validation only.
model.load_state_dict(best_state)
checkpoint_path = Path("paper_experiment/gnn_v2_shallow_s1_dev_split2_best.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "train_subjects": train_subjects,
        "validation_subjects": VAL_SUBJECTS,
        "target_subject": TARGET_SUBJECT,
        "seed": SEED,
    },
    checkpoint_path,
)

print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Checkpoint saved: {checkpoint_path}")
print("Target labels were not used for model selection.")
