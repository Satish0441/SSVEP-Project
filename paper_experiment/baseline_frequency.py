from pathlib import Path
import numpy as np

DATA_DIR = Path("paper_experiment/dataset/0.5s")
TARGET = "S1"

files = sorted(DATA_DIR.glob("S*.npz"))

source_files = [p for p in files if p.stem != TARGET]
target_file = DATA_DIR / f"{TARGET}.npz"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_subject(path):
    with np.load(path) as z:
        X = z["X"]

    # (9,T,40,6) -> (240,9,T)
    X = np.transpose(X, (2, 3, 0, 1))
    X = X.reshape(-1, 9, X.shape[-1])

    y = np.repeat(np.arange(40), 6)

    return X.astype(np.float32), y


# ------------------------------------------------------------
# Build source templates
# ------------------------------------------------------------

templates = []

for path in source_files:
    X, y = load_subject(path)
    templates.append(X)

X_source = np.concatenate(templates, axis=0)

# Class template:
# (40, 9, T)

class_templates = np.zeros(
    (40, 9, X_source.shape[-1]),
    dtype=np.float32
)

for c in range(40):
    class_templates[c] = X_source[
        np.arange(len(X_source)) % 40 == c
    ].mean(axis=0)

# ------------------------------------------------------------
# Target data
# ------------------------------------------------------------

X_target, y_target = load_subject(target_file)

# ------------------------------------------------------------
# Frequency-domain similarity
# ------------------------------------------------------------

def normalize_rows(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / norm


# FFT magnitude for every channel
target_fft = np.abs(np.fft.rfft(X_target, axis=-1))
template_fft = np.abs(np.fft.rfft(class_templates, axis=-1))

# Flatten channels + frequencies
target_flat = target_fft.reshape(len(X_target), -1)
template_flat = template_fft.reshape(40, -1)

target_flat = normalize_rows(target_flat)
template_flat = normalize_rows(template_flat)

# cosine similarity
scores = target_flat @ template_flat.T

pred = scores.argmax(axis=1)

accuracy = np.mean(pred == y_target)

print("=== ZERO-SHOT FREQUENCY TEMPLATE BASELINE ===")
print("Target subject:", TARGET)
print("Source subjects:", len(source_files))
print("Target trials:", len(y_target))
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Chance: 2.50%")
