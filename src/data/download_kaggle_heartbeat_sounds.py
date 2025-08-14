import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env BEFORE importing kaggle so credentials are available at import time
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# Optional: if kaggle.json is missing but env vars exist, create it automatically
def ensure_kaggle_json_from_env():
    import json
    user = os.getenv("aryanbaliyan123")
    key = os.getenv("c69a919d8e63aa82669d68fb3c735e80")
    if not user or not key:
        return
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        kaggle_json.write_text(json.dumps({"username": user, "key": key}))
        try:
            os.chmod(kaggle_json, 0o600)  # best-effort on POSIX; harmless on Windows
        except Exception:
            pass

ensure_kaggle_json_from_env()

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

LABEL_MAP = {
    "normal": "normal",
    "murmur": "murmur",
    "extrasystole": "extrasystole",
    "extrastole": "extrasystole",  # handle alternate spelling if present
    "artifact": "artifact",
    "artifacts": "artifact",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_label_from_path(p: Path) -> Optional[str]:
    parts = [part.lower() for part in p.parts]
    for k in LABEL_MAP.keys():
        if k in parts:
            return LABEL_MAP[k]
    parent = p.parent.name.lower()
    if parent in LABEL_MAP:
        return LABEL_MAP[parent]
    return None

def main():
    project_root = PROJECT_ROOT
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / "kaggle_heartbeat_sounds"
    meta_dir = data_dir / "metadata"
    ensure_dir(raw_dir)
    ensure_dir(meta_dir)

    # Authenticate Kaggle
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Kaggle authentication failed.")
        print("1) Create an API token at https://www.kaggle.com/settings/account")
        print("2) Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars.")
        print(f"Error: {e}")
        sys.exit(1)

    # Download dataset
    dataset = "yasserh/heartbeat-sounds"
    print(f"Downloading Kaggle dataset: {dataset}")
    api.dataset_download_files(dataset, path=str(raw_dir), unzip=True)

    # Scan wav files and build metadata
    wavs = list(raw_dir.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files found under {raw_dir}.")
        sys.exit(1)

    rows = []
    for wav in wavs:
        label = infer_label_from_path(wav)
        if label is None:
            continue
        rows.append({
            "dataset": "kaggle_heartbeat_sounds",
            "source_split": wav.parents[1].name if len(wav.parents) > 1 else "",
            "record_id": wav.stem,
            "filepath": str(wav.resolve()),
            "label_raw": label,
            "label": label,
        })

    if not rows:
        print("No labeled files found. Check the dataset structure.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_csv = meta_dir / "metadata_kaggle_heartbeat_sounds.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} entries to {out_csv}")
    print("Label distribution (Kaggle Heartbeat Sounds):")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()