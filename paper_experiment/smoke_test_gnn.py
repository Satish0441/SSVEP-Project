from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_DIR = Path("paper_experiment/dataset/0.5s")
GRAPH_FILE = Path("paper_experiment/graph_9ch.npz")

TARGET_SUBJECT = "S1"
EPOCHS = 3
BATCH_SIZE = 32
LR = 0.0002

DEVICE = torch.device("cpu")

# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------

graph = np.load(GRAPH_FILE)
A = torch.tensor(graph["adjacency"], dtype=torch.float32)

A_hat = A + torch.eye(A.shape[0])

degree = A_hat.sum(dim=1)
D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))

A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
A_norm = A_norm.to(DEVICE)

# ------------------------------------------------------------
# Load subjects
# ------------------------------------------------------------

files = sorted(DATA_DIR.glob("S*.npz"))

train_files = [
    p for p in files
    if p.stem != TARGET_SUBJECT
]

test_file = DATA_DIR / f"{TARGET_SUBJECT}.npz"

print("Target subject:", TARGET_SUBJECT)
print("Training subjects:", len(train_files))
print("Test subject:", test_file.stem)

# ------------------------------------------------------------
# Convert NPZ into trial samples
#
# Stored:
# X = (9, T, 40, 6)
#
# Convert to:
# samples = (240, 9, T)
# labels  = (240,)
# ------------------------------------------------------------

def load_subject(path):
    with np.load(path) as z:
        X = z["X"]

    # (9, T, 40, 6) -> (40, 6, 9, T)
    X = np.transpose(X, (2, 3, 0, 1))

    # -> (240, 9, T)
    X = X.reshape(-1, 9, X.shape[-1])

    # labels: 0..39, repeated for six blocks
    y = np.repeat(np.arange(40), 6)

    return X.astype(np.float32), y.astype(np.int64)


X_train_list = []
y_train_list = []

for path in train_files:
    X, y = load_subject(path)
    X_train_list.append(X)
    y_train_list.append(y)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

X_test, y_test = load_subject(test_file)

print("Training X:", X_train.shape)
print("Training y:", y_train.shape)
print("Testing X:", X_test.shape)
print("Testing y:", y_test.shape)

# ------------------------------------------------------------
# Source-only normalization
#
# Statistics are calculated ONLY from source subjects.
# ------------------------------------------------------------

mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# ------------------------------------------------------------
# Torch tensors
# ------------------------------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: (B, 9, T)

        B, N, T = x.shape

        x = x.reshape(B * N, 1, T)

        x = self.net(x)

        x = x.reshape(B, N, 32)

        return x


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A_norm):
        # x: (B, N, F)
        x = torch.matmul(A_norm, x)
        x = self.linear(x)
        return x


class SSVEPGNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal = TemporalEncoder()

        self.gcn1 = GCNLayer(32, 64)
        self.gcn2 = GCNLayer(64, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 40)
        )

    def forward(self, x, A_norm):
        x = self.temporal(x)

        x = F.relu(self.gcn1(x, A_norm))

        x = F.relu(self.gcn2(x, A_norm))

        # Global mean pooling over electrodes
        x = x.mean(dim=1)

        return self.classifier(x)


model = SSVEPGNN().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

model.train()

n = len(X_train)

for epoch in range(EPOCHS):

    permutation = torch.randperm(n)

    total_loss = 0.0
    correct = 0

    for start in range(0, n, BATCH_SIZE):

        idx = permutation[start:start + BATCH_SIZE]

        xb = X_train[idx].to(DEVICE)
        yb = y_train[idx].to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb, A_norm)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(idx)

        correct += (
            logits.argmax(dim=1) == yb
        ).sum().item()

    train_acc = correct / n
    avg_loss = total_loss / n

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"loss={avg_loss:.4f} "
        f"train_acc={train_acc:.4f}"
    )

# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(
        X_test.to(DEVICE),
        A_norm
    )

    predictions = logits.argmax(dim=1)

    accuracy = (
        predictions == y_test.to(DEVICE)
    ).float().mean().item()

print()
print("=== SMOKE TEST RESULT ===")
print("Target:", TARGET_SUBJECT)
print("Timing: 0.5s")
print("Test trials:", len(y_test))
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Pipeline complete.")
