import os
import re
import sys
import zipfile
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from tqdm import tqdm
import pandas as pd

PHYSIONET_STATIC_ZIP = "https://physionet.org/static/published-projects/challenge-2016/challenge-2016-1.0.0.zip"

def unify_label(label_raw: str) -> str:
    lr = str(label_raw).strip().lower()
    # Numeric encodings observed in PhysioNet 2016:
    # -1 => normal, 1 => abnormal
    if lr in {"-1", "0", "normal", "n", "norm"}:
        return "normal"
    if lr in {"1", "+1", "abnormal", "abn", "abnorm"}:
        return "abnormal_other"
    # default fallback
    return "abnormal_other"

    
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest: Path, chunk_size: int = 1 << 16):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        total = int(total) if total is not None else None
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if total:
                        pbar.update(len(chunk))

def extract_zip(zip_path: Path, extract_to: Path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

def parse_reference_file(ref_path: Path) -> List[Tuple[str, str]]:
    entries = []
    with open(ref_path, "r", newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                record_id, label_raw = parts[0], parts[1]
                entries.append((record_id, label_raw))
    return entries

def find_reference_file(folder: Path) -> Optional[Path]:
    candidates = list(folder.glob("REFERENCE*"))
    return candidates[0] if candidates else None

def find_training_dirs(root: Path):
    dirs = []
    for d in root.rglob("*"):
        if d.is_dir() and re.fullmatch(r"training-[a-e]", d.name.lower()):
            dirs.append(d)
    return dirs

def build_metadata(training_dirs, out_csv: Path):
    records = []
    for tdir in sorted(training_dirs, key=lambda p: p.name):
        ref_path = find_reference_file(tdir)
        if ref_path is None:
            print(f"WARNING: No REFERENCE file found in {tdir}. Skipping this folder.")
            continue
        ref_entries = parse_reference_file(ref_path)
        for rec_id, label_raw in ref_entries:
            wav_path = (tdir / f"{rec_id}.wav").resolve()
            if not wav_path.exists():
                candidates = [c for c in tdir.glob(f"{rec_id}.*") if c.suffix.lower() == ".wav"]
                if candidates:
                    wav_path = candidates[0].resolve()
            if not wav_path.exists():
                print(f"WARNING: Missing audio for {rec_id} in {tdir.name}")
                continue
            records.append({
                "dataset": "physionet_2016",
                "source_split": tdir.name,
                "record_id": rec_id,
                "filepath": str(wav_path),
                "label_raw": label_raw.strip().lower(),
                "label": unify_label(label_raw),
            })
    if not records:
        print("No records parsed. Check downloads/extractions.")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} entries to {out_csv}")
    print("Label distribution (PhysioNet 2016 unified):")
    print(df["label"].value_counts())

def main(force: bool = False):
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / "physionet_2016"
    zips_dir = raw_dir / "zips"
    extract_dir = raw_dir / "extracted"
    meta_dir = data_dir / "metadata"
    ensure_dir(zips_dir)
    ensure_dir(extract_dir)
    ensure_dir(meta_dir)

    big_zip = zips_dir / "challenge-2016-1.0.0.zip"
    if not big_zip.exists():
        try:
            print(f"Downloading master archive from: {PHYSIONET_STATIC_ZIP}")
            download_file(PHYSIONET_STATIC_ZIP, big_zip)
        except Exception as e:
            print(f"ERROR downloading {PHYSIONET_STATIC_ZIP}: {e}")
            sys.exit(1)
    else:
        print(f"Already downloaded: {big_zip.name}")

    # Check if already extracted
    training_dirs = find_training_dirs(extract_dir)
    if not training_dirs or force:
        print(f"Extracting {big_zip.name} ...")
        extract_zip(big_zip, extract_dir)
        training_dirs = find_training_dirs(extract_dir)

    if not training_dirs:
        print("Could not find any training-[a-e] directories after extraction.")
        print(f"Please check contents of: {extract_dir}")
        sys.exit(1)

    out_csv = meta_dir / "metadata_physionet2016.csv"
    if out_csv.exists() and not force:
        print(f"Metadata already exists: {out_csv}")
        print("Use --force to rebuild it.")
        return

    build_metadata(training_dirs, out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-extract and rebuild metadata")
    args = parser.parse_args()
    main(force=args.force)