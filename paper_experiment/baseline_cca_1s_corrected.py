from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import CCA

DATA_DIR = Path("paper_experiment/dataset/1.0s")
TARGET = "S1"

# Benchmark target frequencies
freqs = np.array([8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,8.4,9.4,10.4,11.4,12.4,13.4,14.4,15.4,8.6,9.6,10.6,11.6,12.6,13.6,14.6,15.6,8.8,9.8,10.8,11.8,12.8,13.8,14.8,15.8])

# Multi-harmonic reference signals
n_harmonics = 3
fs = 250.0

with np.load(DATA_DIR / f"{TARGET}.npz") as z:
    X = z["X"]

# (9,T,40,6) -> (240,9,T)
X = np.transpose(X, (2, 3, 0, 1))
X = X.reshape(-1, 9, X.shape[-1]).astype(np.float64)

y = np.repeat(np.arange(40), 6)

T = X.shape[-1]
t = np.arange(T) / fs


def make_reference(freq):
    refs = []

    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))

    return np.stack(refs, axis=1)


references = [
    make_reference(f)
    for f in freqs
]


def cca_score(eeg, reference):
    """
    Maximum canonical correlation between:
        EEG:       T x channels
        Reference: T x reference_components
    """

    # sklearn CCA expects samples x features
    eeg = eeg.T

    cca = CCA(
        n_components=1,
        max_iter=1000,
        scale=True
    )

    cca.fit(eeg, reference)

    X_c, Y_c = cca.transform(eeg, reference)

    x = X_c[:, 0]
    y = Y_c[:, 0]

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(
        np.sum(x ** 2) *
        np.sum(y ** 2)
    ) + 1e-12

    return abs(np.sum(x * y) / denom)


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

pred = []

for trial_idx in range(len(X)):

    eeg = X[trial_idx]

    trial_scores = []

    for reference in references:
        trial_scores.append(
            cca_score(eeg, reference)
        )

    pred.append(np.argmax(trial_scores))

pred = np.asarray(pred)

accuracy = np.mean(pred == y)

print("=== CLASSICAL MULTI-HARMONIC CCA ===")
print("Target subject:", TARGET)
print("Trials:", len(y))
print("Window: 0.5 s")
print("Channels:", X.shape[1])
print("Harmonics:", n_harmonics)
print("Frequencies:", freqs[0], "to", freqs[-1], "Hz")
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
