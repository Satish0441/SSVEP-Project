from pathlib import Path
import numpy as np
from scipy.io import loadmat

RAW_DIR = Path("data/raw/benchmark")
OUT_DIR = Path("paper_experiment/dataset")

CHANNEL_INDICES = [47, 54, 53, 56, 57, 55, 60, 61, 62]
CHANNEL_NAMES = ["Pz", "PO3", "PO5", "PO4", "PO6", "POz", "O1", "Oz", "O2"]

FS = 250
LATENCY_S = 0.14
STIMULUS_ONSET_S = 0.5
START_SAMPLE = int((STIMULUS_ONSET_S + LATENCY_S) * FS)

SIGNAL_LENGTHS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

EXPECTED_RAW_SHAPE = (64, 1500, 40, 6)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the authoritative Benchmark target mapping.
    freq_phase = loadmat(RAW_DIR / "Freq_Phase.mat")

    freqs = np.asarray(freq_phase["freqs"]).ravel().astype(np.float32)
    phases = np.asarray(freq_phase["phases"]).ravel().astype(np.float32)

    if freqs.shape != (40,):
        raise RuntimeError(f"Expected 40 frequencies, got {freqs.shape}")

    if phases.shape != (40,):
        raise RuntimeError(f"Expected 40 phases, got {phases.shape}")

    subjects = sorted(RAW_DIR.glob("S*.mat"))

    if len(subjects) != 35:
        raise RuntimeError(f"Expected 35 subjects, found {len(subjects)}")

    print("Building paper-protocol dataset")
    print(f"Subjects: {len(subjects)}")
    print(f"Channels: {CHANNEL_NAMES}")
    print(f"Start sample: {START_SAMPLE}")
    print(f"Frequencies: {freqs}")
    print(f"Phases: {phases}")
    print()

    for length in SIGNAL_LENGTHS:
        n_samples = int(length * FS)
        out_dir = OUT_DIR / f"{length:.1f}s"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {length:.1f}s ({n_samples} samples) ===")

        end_sample = START_SAMPLE + n_samples

        for subject_file in subjects:
            subject = subject_file.stem

            raw = loadmat(subject_file)["data"]

            if raw.shape != EXPECTED_RAW_SHAPE:
                raise RuntimeError(
                    f"{subject}: unexpected raw shape {raw.shape}"
                )

            x = raw[
                CHANNEL_INDICES,
                START_SAMPLE:end_sample,
                :,
                :
            ].astype(np.float32)

            expected_shape = (9, n_samples, 40, 6)

            if x.shape != expected_shape:
                raise RuntimeError(
                    f"{subject}: expected {expected_shape}, got {x.shape}"
                )

            if not np.isfinite(x).all():
                raise RuntimeError(f"{subject}: NaN/Inf detected")

            output = out_dir / f"{subject}.npz"

            np.savez_compressed(
                output,
                X=x,
                channel_names=np.array(CHANNEL_NAMES),
                channel_indices=np.array(CHANNEL_INDICES),
                signal_length=np.float32(length),
                sampling_rate=np.int32(FS),
                latency_seconds=np.float32(LATENCY_S),
                stimulus_onset_seconds=np.float32(STIMULUS_ONSET_S),
                start_sample=np.int32(START_SAMPLE),
                freqs=freqs,
                phases=phases,
            )

        print(f"Saved {len(subjects)} subjects -> {out_dir}")


if __name__ == "__main__":
    build()
