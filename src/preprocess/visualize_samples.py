from pathlib import Path
import random

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

SR = 4000
SEGMENT_SECONDS = 5.0

def plot_waveform_and_spec(wav_path: Path, spec_path: Path, start_sec: float, out_png: Path, sr: int = SR):
    # Load original audio and slice the same segment for visualization
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    s = int(start_sec * sr)
    e = s + int(SEGMENT_SECONDS * sr)
    if e > len(y):
        pad = e - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    y_seg = y[s:e]

    S = np.load(spec_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    # Waveform
    t = np.arange(len(y_seg)) / sr
    axes[0].plot(t, y_seg, color="#1f77b4", linewidth=0.8)
    axes[0].set_title(f"Waveform (segment) - {wav_path.name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # Spectrogram
    im = axes[1].imshow(S, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title(f"Log-Mel Spectrogram: {Path(spec_path).name}")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mel bins")
    fig.colorbar(im, ax=axes[1], orientation="vertical", fraction=0.02)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    meta_dir = data_dir / "metadata"
    figs_dir = project_root / "reports" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(meta_dir / "features_index.csv")
    if feats.empty:
        print("features_index.csv is empty. Run extract_features.py first.")
        return

    labels = feats["label"].unique().tolist()
    # Up to 2 samples per label
    for label in labels:
        sub = feats[feats["label"] == label]
        n = min(2, len(sub))
        if n == 0:
            continue
        for sample in sub.sample(n, random_state=42).itertuples():
            wav = Path(sample.source_file)
            spec = Path(sample.feature_path)
            start_sec = float(sample.start_sec)
            out = figs_dir / f"{label}__{wav.stem}__seg{sample.segment_idx}.png"
            plot_waveform_and_spec(wav, spec, start_sec, out)
            print(f"Saved figure: {out}")

if __name__ == "__main__":
    main()