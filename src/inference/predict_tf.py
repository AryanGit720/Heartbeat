import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .audio_utils import (
    load_audio, segment_audio, logmel,
    SR, SEGMENT_SECONDS, HOP_SECONDS
)

class Predictor:
    def __init__(self, model_path: Path, label_map_path: Path, norm_path: Path, abnormal_threshold: float = 0.6):
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        with open(label_map_path, "r") as f:
            lm = json.load(f)
        labels = lm["labels"]
        l2i = {k: int(v) for k, v in lm["label_to_idx"].items()}
        # Make sure idx_to_label aligns with model outputs
        self.idx_to_label = [None] * len(labels)
        for k, v in l2i.items():
            self.idx_to_label[v] = k

        with open(norm_path, "r") as f:
            stats = json.load(f)
        self.mean = float(stats["global_mean"])
        self.std = float(stats["global_std"])
        self.abnormal_threshold = float(abnormal_threshold)

    def _prep_batch(self, feats: List[np.ndarray]) -> np.ndarray:
        # feats: list of [n_mels, T]
        # Normalize and stack; pad to same T if needed
        T_max = max(x.shape[1] for x in feats)
        batch = []
        for x in feats:
            if x.shape[1] < T_max:
                pad = np.zeros((x.shape[0], T_max - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            x = (x - self.mean) / (self.std + 1e-8)
            x = np.expand_dims(x, -1)  # [n_mels, T, 1]
            batch.append(x)
        return np.stack(batch, axis=0)  # [B, n_mels, T, 1]

    def predict_file(self, audio_path: Path) -> Dict:
        y = load_audio(audio_path)
        segs = segment_audio(y, sr=SR, seg_sec=SEGMENT_SECONDS, hop_sec=HOP_SECONDS)
        if not segs:
            raise ValueError("No segments produced from audio; check the file.")

        # Compute features for each segment
        feats = []
        for (s, e, s_sec, e_sec) in segs:
            y_seg = y[s:e]
            feats.append(logmel(y_seg, sr=SR))

        batch = self._prep_batch(feats)
        probs = self.model.predict(batch, verbose=0)  # [B, C]
        probs = probs.astype(np.float32)

        # Aggregate across segments (average probs)
        rec_probs = probs.mean(axis=0)
        pred_idx = int(np.argmax(rec_probs))
        pred_label = self.idx_to_label[pred_idx]
        confidence = float(rec_probs[pred_idx])

        # Segment-level details
        seg_details = []
        for i, ((s, e, s_sec, e_sec), p) in enumerate(zip(segs, probs)):
            top_i = int(np.argmax(p))
            top_label = self.idx_to_label[top_i]
            seg_details.append({
                "segment_idx": i,
                "start_sec": float(s_sec),
                "end_sec": float(e_sec),
                "top_label": top_label,
                "probs": {self.idx_to_label[j]: float(p[j]) for j in range(len(self.idx_to_label))}
            })

        # Highlight segments likely abnormal (if abnormal class exists)
        highlight_idxs = []
        if "abnormal_other" in self.idx_to_label:
            ab_idx = self.idx_to_label.index("abnormal_other")
            for i, p in enumerate(probs):
                if p[ab_idx] >= self.abnormal_threshold:
                    highlight_idxs.append(i)

        return {
            "record": {
                "primary_prediction": pred_label,
                "confidence": confidence,
                "probs": {self.idx_to_label[j]: float(rec_probs[j]) for j in range(len(self.idx_to_label))}
            },
            "segments": seg_details,
            "segment_seconds": {"length": SEGMENT_SECONDS, "hop": HOP_SECONDS},
            "highlight_segments": highlight_idxs,
        }