from pathlib import Path
import numpy as np

root = Path(r"paper_experiment\dataset")

lengths = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

expected_channels = [
    "Pz", "PO3", "PO5", "PO4", "PO6",
    "POz", "O1", "Oz", "O2"
]

problems = []

print("=== DATASET INTEGRITY ===")

for length in lengths:
    folder = root / f"{length:.1f}s"
    files = sorted(folder.glob("S*.npz"))

    expected_shape = (9, int(length * 250), 40, 6)

    print(f"{length:.1f}s: {len(files)} files")

    if len(files) != 35:
        problems.append((f"{length:.1f}s", "file_count", len(files)))

    for path in files:
        with np.load(path) as z:
            X = z["X"]

            if X.shape != expected_shape:
                problems.append(
                    (f"{length:.1f}s", path.name, "shape", X.shape)
                )

            if not np.isfinite(X).all():
                problems.append(
                    (f"{length:.1f}s", path.name, "nonfinite")
                )

            if z["channel_names"].tolist() != expected_channels:
                problems.append(
                    (f"{length:.1f}s", path.name, "channels")
                )

            if int(z["start_sample"]) != 160:
                problems.append(
                    (f"{length:.1f}s", path.name, "start_sample",
                     int(z["start_sample"]))
                )

print()
print("Problems:", problems)
print(
    "ALL DATASET CHECKS PASS:",
    len(problems) == 0
)
