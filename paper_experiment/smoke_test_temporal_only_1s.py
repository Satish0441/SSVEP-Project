from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path("paper_experiment/dataset/1.0s")

TARGET_SUBJECT = "S1"
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.0002

DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def load_subject(path):

    with np.load(path) as z:
        X = z["X"]

    # (9,T,40,6) -> (40,6,9,T)
    X = np.transpose(X, (2, 3, 0, 1))

    # -> (240,9,T)
    X = X.reshape(-1, 9, X.shape[-1])

    y = np.repeat(np.arange(40), 6)

    return X.astype(np.float32), y.astype(np.int64)


files = sorted(DATA_DIR.glob("S*.npz"))

train_files = [
    p for p in files
    if p.stem != TARGET_SUBJECT
]

test_file = DATA_DIR / f"{TARGET_SUBJECT}.npz"

X_train_list = []
y_train_list = []

for path in train_files:
    X, y = load_subject(path)
    X_train_list.append(X)
    y_train_list.append(y)

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)

X_test, y_test = load_subject(test_file)

print("Target subject:", TARGET_SUBJECT)
print("Training subjects:", len(train_files))
print("Training X:", X_train.shape)
print("Testing X:", X_test.shape)


# ------------------------------------------------------------
# Source-only normalization
# ------------------------------------------------------------

mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# ------------------------------------------------------------
# Temporal encoder
# ------------------------------------------------------------

class TemporalEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(
                1, 16,
                kernel_size=9,
                padding=4
            ),

            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(
                16, 32,
                kernel_size=9,
                padding=4
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(
                32, 32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(8)
        )

    def forward(self, x):

        B, N, T = x.shape

        x = x.reshape(B * N, 1, T)

        x = self.conv(x)

        # B,N,256
        x = x.reshape(B, N, 32 * 8)

        return x


# ------------------------------------------------------------
# Temporal-only classifier
# ------------------------------------------------------------

class TemporalOnly(nn.Module):

    def __init__(self):

        super().__init__()

        self.temporal = TemporalEncoder()

        self.projection = nn.Linear(
            32 * 8,
            128
        )

        self.classifier = nn.Sequential(

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 40)
        )

    def forward(self, x):

        x = self.temporal(x)

        x = F.relu(
            self.projection(x)
        )

        # Mean over the 9 electrodes.
        x = x.mean(dim=1)

        return self.classifier(x)


model = TemporalOnly().to(DEVICE)

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

        idx = permutation[
            start:start + BATCH_SIZE
        ]

        xb = X_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(
            logits,
            yb
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(idx)

        correct += (
            logits.argmax(dim=1) == yb
        ).sum().item()

    train_acc = correct / n
    avg_loss = total_loss / n

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"loss={avg_loss:.4f} "
        f"train_acc={train_acc * 100:.2f}%"
    )


# ------------------------------------------------------------
# Target test
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(X_test)

    predictions = logits.argmax(dim=1)

    accuracy = (
        predictions == y_test
    ).float().mean().item()

print()
print("=== TEMPORAL-ONLY ZERO-SHOT TEST ===")
print("Target:", TARGET_SUBJECT)
print("Timing: 1.0s")
print("Test trials:", len(y_test))
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Chance: 2.50%")
