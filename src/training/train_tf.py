import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

from .model_tf import build_model


AUTOTUNE = tf.data.AUTOTUNE


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_norm(meta_dir: Path) -> Tuple[float, float]:
    norm_path = meta_dir / "feature_norm.json"
    with open(norm_path, "r") as f:
        stats = json.load(f)
    return float(stats["global_mean"]), float(stats["global_std"])


def load_label_map(meta_dir: Path) -> Tuple[List[str], Dict[str, int], List[str]]:
    with open(meta_dir / "label_map.json", "r") as f:
        lm = json.load(f)
    labels = lm["labels"]
    label_to_idx = lm["label_to_idx"]
    # idx_to_label as list where index corresponds to class index
    idx_to_label = [None] * len(labels)
    for k, v in label_to_idx.items():
        idx_to_label[v] = k
    return labels, label_to_idx, idx_to_label


def determine_input_shape(train_df: pd.DataFrame) -> Tuple[int, int]:
    """Load one feature to infer (n_mels, time_steps)."""
    sample_path = Path(train_df.iloc[0]["feature_path"])
    x = np.load(sample_path)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature, got shape {x.shape} at {sample_path}")
    n_mels, time_steps = x.shape
    return n_mels, time_steps


def spec_augment(x: tf.Tensor, max_freq_mask: int = 8, max_time_mask: int = 16) -> tf.Tensor:
    """
    Simple SpecAugment: randomly mask a block in frequency and time.
    x: [n_mels, T, 1]
    """
    n_mels = tf.shape(x)[0]
    n_time = tf.shape(x)[1]

    # Frequency mask
    def apply_freq_mask(x):
        f = tf.random.uniform((), minval=0, maxval=max_freq_mask + 1, dtype=tf.int32)
        f0 = tf.cond(
            n_mels > f,
            lambda: tf.random.uniform((), minval=0, maxval=n_mels - f + 1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        left = x[:f0, :, :]
        mid = tf.zeros_like(x[f0 : f0 + f, :, :])
        right = x[f0 + f :, :, :]
        return tf.concat([left, mid, right], axis=0)

    # Time mask
    def apply_time_mask(x):
        t = tf.random.uniform((), minval=0, maxval=max_time_mask + 1, dtype=tf.int32)
        t0 = tf.cond(
            n_time > t,
            lambda: tf.random.uniform((), minval=0, maxval=n_time - t + 1, dtype=tf.int32),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        left = x[:, :t0, :]
        mid = tf.zeros_like(x[:, t0 : t0 + t, :])
        right = x[:, t0 + t :, :]
        return tf.concat([left, mid, right], axis=1)

    # Apply with 0.5 probability each
    x = tf.cond(tf.random.uniform(()) < 0.5, lambda: apply_freq_mask(x), lambda: x)
    x = tf.cond(tf.random.uniform(()) < 0.5, lambda: apply_time_mask(x), lambda: x)
    return x


def make_dataset(
    df: pd.DataFrame,
    batch_size: int,
    n_mels: int,
    time_steps: int,
    mean: float,
    std: float,
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    paths = df["feature_path"].tolist()
    labels = df["label_idx"].astype(int).tolist()

    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t = tf.constant(std, dtype=tf.float32)

    def _load_fn(path_str):
        path = path_str.decode("utf-8")
        x = np.load(path).astype(np.float32)  # [n_mels, T]
        # pad/truncate along time axis if needed
        if x.shape[1] < time_steps:
            pad = np.zeros((x.shape[0], time_steps - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > time_steps:
            x = x[:, :time_steps]
        # normalize (global mean/std)
        x = (x - mean) / (std + 1e-8)
        x = np.expand_dims(x, axis=-1)  # [n_mels, T, 1]
        return x

    def _tf_load(path, label):
        x = tf.numpy_function(_load_fn, [path], tf.float32)
        x.set_shape((n_mels, time_steps, 1))
        if augment:
            x = spec_augment(x)
        return x, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(paths)), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _tf_load(p, y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def compute_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    counts = train_df["label_idx"].value_counts().to_dict()
    total = sum(counts.values())
    num_classes = len(counts)
    class_weight = {int(idx): total / (num_classes * int(cnt)) for idx, cnt in counts.items()}
    return class_weight


def ensure_dirs(project_root: Path) -> Tuple[Path, Path]:
    models_dir = project_root / "models" / "tf_heart_sound"
    logs_dir = project_root / "logs" / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, logs_dir


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: Path, normalize: bool = False):
    plt.figure(figsize=(6, 5))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve_binary(y_true: np.ndarray, y_prob_pos: np.ndarray, pos_label_name: str, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    auc = roc_auc_score(y_true, y_prob_pos)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({pos_label_name})")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true", help="Disable SpecAugment for training")
    args = parser.parse_args()

    set_seeds(42)

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    meta_dir = data_dir / "metadata"
    figs_dir = project_root / "reports" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    models_dir, logs_dir = ensure_dirs(project_root)

    # Load metadata
    feats = pd.read_csv(meta_dir / "features_index.csv")
    labels, label_to_idx, idx_to_label = load_label_map(meta_dir)
    mean, std = load_norm(meta_dir)

    # Split
    train_df = feats[feats["split"] == "train"].reset_index(drop=True)
    val_df = feats[feats["split"] == "val"].reset_index(drop=True)
    test_df = feats[feats["split"] == "test"].reset_index(drop=True)

    print("Counts by split:")
    print(train_df["label"].value_counts(), "\n")
    print(val_df["label"].value_counts(), "\n")
    print(test_df["label"].value_counts(), "\n")

    if len(labels) < 2:
        raise RuntimeError("Need at least 2 classes to train.")

    # Input shape
    n_mels, time_steps = determine_input_shape(train_df)
    print(f"Input shape: n_mels={n_mels}, time_steps={time_steps}")

    # Datasets
    train_ds = make_dataset(
        train_df, args.batch_size, n_mels, time_steps, mean, std,
        augment=not args.no_augment, shuffle=True
    )
    val_ds = make_dataset(
        val_df, args.batch_size, n_mels, time_steps, mean, std,
        augment=False, shuffle=False
    )
    test_ds = make_dataset(
        test_df, args.batch_size, n_mels, time_steps, mean, std,
        augment=False, shuffle=False
    )

    # Class weights for imbalance
    class_weight = compute_class_weights(train_df)
    print("Class weights:", class_weight)

    # Build model
    model = build_model(n_mels=n_mels, num_classes=len(labels), dropout=args.dropout, lr=float(args.lr))

    # Callbacks
    ckpt_path = models_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2), verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir)),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # Save final model too
    final_path = models_dir / "final.keras"
    model.save(final_path)
    print(f"Saved best to {ckpt_path} and final to {final_path}")

    # Evaluation (per-segment)
    print("\nEvaluating on test (per-segment)...")
    probs = model.predict(test_ds, verbose=1)
    y_true = test_df["label_idx"].to_numpy()
    y_pred = np.argmax(probs, axis=1)

    seg_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plot_confusion_matrix(seg_cm, labels=idx_to_label,
                          title="Confusion Matrix (segments)", out_path=figs_dir / "cm_segments.png", normalize=False)
    plot_confusion_matrix(seg_cm, labels=idx_to_label,
                          title="Confusion Matrix (segments, normalized)", out_path=figs_dir / "cm_segments_norm.png", normalize=True)

    print("\nClassification report (segments):")
    print(classification_report(y_true, y_pred, target_names=idx_to_label, digits=4))

    # ROC (binary only)
    if len(labels) == 2:
        # Choose positive class as 'abnormal_other' if present
        pos_name = "abnormal_other" if "abnormal_other" in idx_to_label else idx_to_label[1]
        pos_idx = idx_to_label.index(pos_name)
        y_true_bin = (y_true == pos_idx).astype(int)
        y_prob_pos = probs[:, pos_idx]
        roc_auc = roc_auc_score(y_true_bin, y_prob_pos)
        print(f"ROC AUC (positive='{pos_name}'): {roc_auc:.4f}")
        plot_roc_curve_binary(y_true_bin, y_prob_pos, pos_name, figs_dir / "roc_curve.png")

    # Save per-segment predictions CSV
    seg_pred_df = test_df.copy()
    for i, name in enumerate(idx_to_label):
        seg_pred_df[f"prob_{name}"] = probs[:, i]
    seg_pred_df["pred_idx"] = y_pred
    seg_pred_df["pred_label"] = [idx_to_label[i] for i in y_pred]
    seg_pred_df.to_csv(meta_dir / "test_predictions_segments.csv", index=False)
    print(f"Saved per-segment predictions to: {meta_dir / 'test_predictions_segments.csv'}")

    # Record-level aggregation: average probs across segments of the same record_id
    print("\nAggregating by record_id (average probs)...")
    tmp = seg_pred_df.groupby("record_id")
    rec_probs = tmp[[f"prob_{name}" for name in idx_to_label]].mean()
    rec_true = tmp["label_idx"].first()  # all segments have same label
    rec_pred_idx = np.argmax(rec_probs.to_numpy(), axis=1)
    rec_cm = confusion_matrix(rec_true.to_numpy(), rec_pred_idx, labels=list(range(len(labels))))

    plot_confusion_matrix(rec_cm, labels=idx_to_label,
                          title="Confusion Matrix (records)", out_path=figs_dir / "cm_records.png", normalize=False)
    plot_confusion_matrix(rec_cm, labels=idx_to_label,
                          title="Confusion Matrix (records, normalized)", out_path=figs_dir / "cm_records_norm.png", normalize=True)

    print("\nClassification report (records):")
    print(classification_report(rec_true, rec_pred_idx, target_names=idx_to_label, digits=4))

    # Save record-level predictions CSV
    rec_out = rec_probs.copy()
    rec_out["true_idx"] = rec_true
    rec_out["true_label"] = [idx_to_label[i] for i in rec_true]
    rec_out["pred_idx"] = rec_pred_idx
    rec_out["pred_label"] = [idx_to_label[i] for i in rec_pred_idx]
    rec_out.to_csv(meta_dir / "test_predictions_records.csv")
    print(f"Saved record-level predictions to: {meta_dir / 'test_predictions_records.csv'}")


if __name__ == "__main__":
    main()