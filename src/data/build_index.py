from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]
    meta_dir = project_root / "data" / "metadata"

    candidates = [
        meta_dir / "metadata_physionet2016.csv",
        meta_dir / "metadata_manual_multiclass.csv",  # optional, if you used the manual 4-class dataset
        # meta_dir / "metadata_circor.csv",  # we can add this later if you decide to ingest CirCor
    ]

    dfs = [pd.read_csv(p) for p in candidates if p.exists()]
    if not dfs:
        print("No metadata files found. Run the download/ingest scripts first.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Allowed labels for now
    target_labels = {"normal", "murmur", "extrasystole", "artifact", "abnormal_other"}
    df = df[df["label"].isin(target_labels)].copy()

    out_csv = meta_dir / "dataset_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"Unified dataset index saved to: {out_csv}")
    print("Overall label distribution:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()