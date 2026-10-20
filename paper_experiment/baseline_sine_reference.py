from pathlib import Path
import numpy as np

DATA_DIR = Path("paper_experiment/dataset/0.5s")
TARGET = "S1"

# Use the authoritative target order saved with the dataset.
with np.load(DATA_DIR / f"{TARGET}.npz") as z:
    X = z["X"]
    freqs = z["freqs"].astype(np.float64)

# (9,T,40,6) -> (240,9,T)
X = np.transpose(X, (2, 3, 0, 1))
X = X.reshape(-1, 9, X.shape[-1]).astype(np.float64)

y = np.repeat(np.arange(40), 6)

T = X.shape[-1]
fs = 250.0
t = np.arange(T) / fs

scores = np.zeros((len(X), 40))

for c, f in enumerate(freqs):

    sin_ref = np.sin(2 * np.pi * f * t)
    cos_ref = np.cos(2 * np.pi * f * t)

    sin_ref /= np.linalg.norm(sin_ref)
    cos_ref /= np.linalg.norm(cos_ref)

    Xc = X - X.mean(axis=-1, keepdims=True)

    sin_score = np.sum(Xc * sin_ref, axis=-1)
    cos_score = np.sum(Xc * cos_ref, axis=-1)

    channel_score = np.sqrt(sin_score**2 + cos_score**2)

    scores[:, c] = channel_score.mean(axis=1)

pred = scores.argmax(axis=1)

accuracy = np.mean(pred == y)

print("=== ZERO-SHOT SIN/COS REFERENCE BASELINE ===")
print("Target subject:", TARGET)
print("Trials:", len(y))
print("Window: 0.5 s")
print("Channels:", 9)
print("Frequencies:", freqs[0], "to", freqs[-1], "Hz")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Chance: 2.50%")

print("\nPer-target accuracy:")
for c in range(40):
    mask = y == c
    acc = np.mean(pred[mask] == y[mask])
    print(f"{c+1:02d}  {freqs[c]:4.1f} Hz  {acc*100:6.2f}%")
