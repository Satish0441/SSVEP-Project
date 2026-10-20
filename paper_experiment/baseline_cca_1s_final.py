from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import CCA

DATA_DIR = Path("paper_experiment/dataset/1.0s")
TARGET = "S1"

with np.load(DATA_DIR / f"{TARGET}.npz") as z:
    X = z["X"]
    freqs = z["freqs"].astype(np.float64)

# (9,T,40,6) -> (240,9,T)
X = np.transpose(X, (2, 3, 0, 1))
X = X.reshape(-1, 9, X.shape[-1]).astype(np.float64)

y = np.repeat(np.arange(40), 6)

fs = 250.0
T = X.shape[-1]
t = np.arange(T) / fs

n_harmonics = 3


def make_reference(freq):
    refs = []

    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))

    return np.stack(refs, axis=1)


references = [make_reference(f) for f in freqs]


def cca_score(eeg, reference):
    eeg = eeg.T

    cca = CCA(
        n_components=1,
        max_iter=1000,
        scale=True
    )

    cca.fit(eeg, reference)

    X_c, Y_c = cca.transform(eeg, reference)

    x = X_c[:, 0]
    y_ref = Y_c[:, 0]

    x -= x.mean()
    y_ref -= y_ref.mean()

    denom = np.sqrt(
        np.sum(x ** 2) *
        np.sum(y_ref ** 2)
    ) + 1e-12

    return abs(np.sum(x * y_ref) / denom)


pred = []

for trial_idx in range(len(X)):

    eeg = X[trial_idx]

    scores = [
        cca_score(eeg, reference)
        for reference in references
    ]

    pred.append(np.argmax(scores))

pred = np.asarray(pred)

accuracy = np.mean(pred == y)

print("=== CLASSICAL MULTI-HARMONIC CCA ===")
print("Target subject:", TARGET)
print("Trials:", len(y))
print("Window:", float(np.load(DATA_DIR / f"{TARGET}.npz")["signal_length"]), "s")
print("Channels:", X.shape[1])
print("Harmonics:", n_harmonics)
print("Frequency mapping loaded from dataset")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Chance: 2.50%")

print("\nPer-target accuracy:")

for c in range(40):
    mask = y == c
    acc = np.mean(pred[mask] == y[mask])

    print(
        f"{c+1:02d}  "
        f"{freqs[c]:4.1f} Hz  "
        f"{acc*100:6.2f}%"
    )
