import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

AUTOTUNE = tf.data.AUTOTUNE

def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_norm(meta_dir: Path) -> Tuple[float, float]:
    with open(meta_dir / "feature_norm.json", "r") as f:
        stats = json.load(f)
    return float(stats["global_mean"]), float(stats["global_std"])

def load_label_map(meta_dir: Path) -> Tuple[List[str], Dict[str, int], List[str]]:
    with open(meta_dir / "label_map.json", "r") as f:
        lm = json.load(f)
    labels = lm["labels"]
    l2i = {k: int(v) for k, v in lm["label_to_idx"].items()}
    idx_to_label = [None] * len(labels)
    for k, v in l2i.items():
        idx_to_label[v] = k
    return labels, l2i, idx_to_label

def determine_input_shape(feats_df: pd.DataFrame) -> Tuple[int, int]:
    # Prefer a train sample to match training padding, fallback to any
    df = feats_df
    if "split" in df.columns and (df["split"] == "train").any():
        df = df[df["split"] == "train"]
    sample_path = Path(df.iloc[0]["feature_path"])
    x = np.load(sample_path)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature, got {x.shape} at {sample_path}")
    n_mels, time_steps = x.shape
    return n_mels, time_steps

def make_dataset(df: pd.DataFrame, batch_size: int, n_mels: int, time_steps: int, mean: float, std: float) -> tf.data.Dataset:
    paths = df["feature_path"].tolist()
    labels = df["label_idx"].astype(int).tolist()

    def _load_fn(path_str):
        p = path_str.decode("utf-8")
        x = np.load(p).astype(np.float32)  # [n_mels, T]
        if x.shape[1] < time_steps:
            pad = np.zeros((x.shape[0], time_steps - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > time_steps:
            x = x[:, :time_steps]
        x = (x - mean) / (std + 1e-8)
        x = np.expand_dims(x, axis=-1)  # [n_mels, T, 1]
        return x

    def _tf_load(path, label):
        x = tf.numpy_function(_load_fn, [path], tf.float32)
        x.set_shape((n_mels, time_steps, 1))
        return x, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _tf_load(p, y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: Path, normalize: bool = False):
    plt.figure(figsize=(6, 5))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_roc_curve_binary(y_true: np.ndarray, y_prob_pos: np.ndarray, pos_label_name: str, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    auc = roc_auc_score(y_true, y_prob_pos)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({pos_label_name})"); plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/tf_heart_sound/best.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seeds(42)
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    meta_dir = data_dir / "metadata"
    figs_dir = project_root / "reports" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(meta_dir / "features_index.csv")
    test_df = feats[feats["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        print("No test data found in features_index.csv")
        return

    labels, label_to_idx, idx_to_label = load_label_map(meta_dir)
    mean, std = load_norm(meta_dir)
    n_mels, time_steps = determine_input_shape(feats)

    test_ds = make_dataset(test_df, args.batch_size, n_mels, time_steps, mean, std)

    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    print("Predicting on test set...")
    probs = model.predict(test_ds, verbose=1)
    y_true = test_df["label_idx"].to_numpy()
    y_pred = np.argmax(probs, axis=1)

    # Segment-level metrics
    seg_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plot_confusion_matrix(seg_cm, idx_to_label, "Confusion Matrix (segments)", figs_dir / "cm_segments.png", normalize=False)
    plot_confusion_matrix(seg_cm, idx_to_label, "Confusion Matrix (segments, normalized)", figs_dir / "cm_segments_norm.png", normalize=True)
    print("\nClassification report (segments):")
    print(classification_report(y_true, y_pred, target_names=idx_to_label, digits=4))

    # ROC (binary)
    if len(labels) == 2:
        pos_name = "abnormal_other" if "abnormal_other" in idx_to_label else idx_to_label[1]
        pos_idx = idx_to_label.index(pos_name)
        y_true_bin = (y_true == pos_idx).astype(int)
        y_prob_pos = probs[:, pos_idx]
        roc_auc = roc_auc_score(y_true_bin, y_prob_pos)
        print(f"ROC AUC (positive='{pos_name}'): {roc_auc:.4f}")
        plot_roc_curve_binary(y_true_bin, y_prob_pos, pos_name, figs_dir / "roc_curve.png")

    # Save per-segment predictions
    seg_pred_df = test_df.copy()
    for i, name in enumerate(idx_to_label):
        seg_pred_df[f"prob_{name}"] = probs[:, i]
    seg_pred_df["pred_idx"] = y_pred
    seg_pred_df["pred_label"] = [idx_to_label[i] for i in y_pred]
    seg_pred_df.to_csv(meta_dir / "test_predictions_segments.csv", index=False)
    print(f"Saved per-segment predictions to: {meta_dir / 'test_predictions_segments.csv'}")

    # Record-level aggregation (average probs)
    tmp = seg_pred_df.groupby("record_id")
    rec_probs = tmp[[f"prob_{name}" for name in idx_to_label]].mean()
    rec_true = tmp["label_idx"].first()
    rec_pred_idx = np.argmax(rec_probs.to_numpy(), axis=1)
    rec_cm = confusion_matrix(rec_true.to_numpy(), rec_pred_idx, labels=list(range(len(labels))))
    plot_confusion_matrix(rec_cm, idx_to_label, "Confusion Matrix (records)", figs_dir / "cm_records.png", normalize=False)
    plot_confusion_matrix(rec_cm, idx_to_label, "Confusion Matrix (records, normalized)", figs_dir / "cm_records_norm.png", normalize=True)
    print("\nClassification report (records):")
    print(classification_report(rec_true, rec_pred_idx, target_names=idx_to_label, digits=4))

    # Save record-level predictions
    rec_out = rec_probs.copy()
    rec_out["true_idx"] = rec_true
    rec_out["true_label"] = [idx_to_label[i] for i in rec_true]
    rec_out["pred_idx"] = rec_pred_idx
    rec_out["pred_label"] = [idx_to_label[i] for i in rec_pred_idx]
    rec_out.to_csv(meta_dir / "test_predictions_records.csv")
    print(f"Saved record-level predictions to: {meta_dir / 'test_predictions_records.csv'}")

if __name__ == "__main__":
    main()