import json
import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Audio + feature params
SR = 4000
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
FMIN = 20.0
FMAX = 2000.0

# Segmentation
SEGMENT_SECONDS = 5.0
HOP_SECONDS = 2.5  # 50% overlap

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_audio_resample(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    # Remove DC offset and clip extremes
    if len(y) == 0:
        return np.zeros(int(SEGMENT_SECONDS * sr), dtype=np.float32)
    y = y - np.mean(y)
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)

def segment_audio(y: np.ndarray, sr: int, seg_sec: float, hop_sec: float) -> List[Tuple[int, int, float]]:
    seg_len = int(seg_sec * sr)
    hop_len = int(hop_sec * sr)
    if len(y) < seg_len:
        pad = seg_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    starts = list(range(0, max(1, len(y) - seg_len + 1), hop_len))
    segments = [(s, s + seg_len, s / sr) for s in starts]
    return segments

def logmel_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=1.0)  # log power
    return S_db.astype(np.float32)

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    meta_dir = data_dir / "metadata"
    processed_dir = data_dir / "processed" / "logmels"
    ensure_dir(processed_dir)

    splits = pd.read_csv(meta_dir / "splits.csv")
    with open(meta_dir / "label_map.json", "r") as f:
        label_info = json.load(f)
    label_to_idx: Dict[str, int] = label_info["label_to_idx"]

    rows = []
    # Global normalization stats over TRAIN segments only
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for split_name in ["train", "val", "test"]:
        split_df = splits[splits["split"] == split_name].reset_index(drop=True)
        out_dir = processed_dir / split_name
        ensure_dir(out_dir)
        print(f"\nProcessing {split_name}: {len(split_df)} files")

        for _, r in tqdm(split_df.iterrows(), total=len(split_df)):
            wav_path = Path(r["filepath"])
            label = r["label"]
            try:
                y = load_audio_resample(wav_path, SR)
                segments = segment_audio(y, SR, SEGMENT_SECONDS, HOP_SECONDS)
                if not segments:
                    continue
                for i, (s, e, start_sec) in enumerate(segments):
                    y_seg = y[s:e]
                    feat = logmel_from_audio(y_seg, SR)  # [n_mels, T]
                    rec_id = f"{Path(r['record_id']).name}_{i}"
                    out_path = out_dir / f"{rec_id}.npy"
                    np.save(out_path, feat)

                    if split_name == "train":
                        X = feat.reshape(-1)  # flatten
                        total_sum += float(X.sum())
                        total_sumsq += float((X ** 2).sum())
                        total_count += int(X.size)

                    rows.append({
                        "split": split_name,
                        "record_id": r["record_id"],
                        "segment_idx": i,
                        "start_sec": round(start_sec, 3),
                        "feature_path": str(out_path.resolve()),
                        "label": label,
                        "label_idx": label_to_idx[label],
                        "source_file": str(wav_path.resolve()),
                    })
            except Exception as ex:
                print(f"ERROR processing {wav_path}: {ex}")
                traceback.print_exc()

    features_df = pd.DataFrame(rows)
    feat_index_path = meta_dir / "features_index.csv"
    features_df.to_csv(feat_index_path, index=False)
    print(f"\nSaved feature index to: {feat_index_path}")
    print("Counts by split and label:")
    if not features_df.empty:
        print(features_df.groupby(['split', 'label']).size())

    # Save global normalization stats
    if total_count > 0:
        global_mean = total_sum / total_count
        variance = max(0.0, (total_sumsq / total_count) - (global_mean ** 2))
        global_std = float(np.sqrt(variance + 1e-12))
        norm_path = meta_dir / "feature_norm.json"
        with open(norm_path, "w") as f:
            json.dump({"global_mean": float(global_mean), "global_std": global_std}, f, indent=2)
        print(f"Saved normalization stats to: {norm_path}")

if __name__ == "__main__":
    main()