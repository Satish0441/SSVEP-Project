from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import CCA

DATA_DIR = Path("paper_experiment/dataset/1.0s")
TARGET = "S1"
# Use the authoritative target order saved with the dataset.
with np.load(DATA_DIR / f"{TARGET}.npz") as z:
    X = z["X"]
    freqs = z["freqs"].astype(np.float64)

# Multi-harmonic reference signals
n_harmonics = 3
fs = 250.0


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
print("Window: 1.0 s")
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

print("\nDiagnostic: first trial's top 5 predicted frequencies")

test_eeg = X[0]
scores = np.array([
    cca_score(test_eeg, ref)
    for ref in references
])

top5 = np.argsort(scores)[-5:][::-1]

for idx in top5:
    print(
        f"class={idx:02d}, "
        f"frequency={freqs[idx]:.2f} Hz, "
        f"score={scores[idx]:.4f}"
    )

print("True class:", y[0])
print("True frequency:", f"{freqs[y[0]]:.2f} Hz")


print("\nDiagnostic: prediction counts by true class")
for c in range(40):
    mask = y == c
    counts = np.bincount(pred[mask], minlength=40)
    top = np.argsort(counts)[-3:][::-1]

    print(
        f"true={c:02d} ({freqs[c]:4.1f} Hz): "
        + ", ".join(
            f"pred={k:02d} ({freqs[k]:4.1f} Hz): {counts[k]}"
            for k in top if counts[k] > 0
        )
    )

