from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import CCA

DATA_DIR = Path("paper_experiment/dataset/0.5s")
TARGET = "S1"
# Use the authoritative target order saved with the dataset.
with np.load(DATA_DIR / f"{TARGET}.npz") as z:
    X = z["X"]
    freqs = z["freqs"].astype(np.float64)
    phases = z["phases"].astype(np.float64)

# Multi-harmonic reference signals
n_harmonics = 3
fs = 250.0


# (9,T,40,6) -> (240,9,T)
X = np.transpose(X, (2, 3, 0, 1))
X = X.reshape(-1, 9, X.shape[-1]).astype(np.float64)

y = np.repeat(np.arange(40), 6)

T = X.shape[-1]
t = np.arange(T) / fs


def make_reference(freq, phase):
    refs = []

    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t + h * phase))
        refs.append(np.cos(2 * np.pi * h * freq * t + h * phase))

    return np.stack(refs, axis=1)

references = [
    make_reference(f, p)
    for f, p in zip(freqs, phases)
]

def cca_score(eeg, reference):
    """
    Standard SSVEP CCA score using the largest canonical
    correlation from whitened covariance matrices.

    eeg:       (channels, samples)
    reference: (samples, reference_components)
    """
    Xc = eeg.T.astype(np.float64)
    Yc = reference.astype(np.float64)

    # Center both matrices
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    Yc = Yc - Yc.mean(axis=0, keepdims=True)

    # Covariance matrices
    Cxx = Xc.T @ Xc
    Cyy = Yc.T @ Yc
    Cxy = Xc.T @ Yc

    # Symmetric inverse square root with numerical stabilization
    def inv_sqrt(C):
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

    Wx = inv_sqrt(Cxx)
    Wy = inv_sqrt(Cyy)

    # Canonical correlations are singular values
    M = Wx @ Cxy @ Wy
    singular_values = np.linalg.svd(M, compute_uv=False)

    return float(np.clip(singular_values[0], 0.0, 1.0))


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

