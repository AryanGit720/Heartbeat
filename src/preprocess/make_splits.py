import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    project_root = Path(__file__).resolve().parents[2]
    meta_dir = project_root / "data" / "metadata"
    index_csv = meta_dir / "dataset_index.csv"
    out_path = meta_dir / "splits.csv"

    df = pd.read_csv(index_csv)

    counts = df["label"].value_counts()
    print("Label distribution in dataset_index.csv:")
    print(counts)

    if counts.shape[0] < 2:
        print("\nERROR: Need at least 2 classes to proceed. Current labels:", list(counts.index))
        return

    # Stratified 70/15/15 split
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1765, stratify=train_df["label"], random_state=42
    )
    # 0.1765 of 0.85 ≈ 0.15, so final ≈ 70/15/15

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"

    splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    splits.to_csv(out_path, index=False)
    print(f"\nSaved splits to: {out_path}")
    print("Split counts:")
    print(splits.groupby(["split", "label"]).size())

    # Save label map for training
    labels = sorted(splits["label"].unique())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    with open(meta_dir / "label_map.json", "w") as f:
        json.dump({"labels": labels, "label_to_idx": label_to_idx}, f, indent=2)
    print("\nLabel map:", label_to_idx)

if __name__ == "__main__":
    main()