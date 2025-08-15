# docs/make_docs_placeholders.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

def savefig(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p.relative_to(Path.cwd()) if Path.cwd() in p.parents else p}")

def upload_placeholder():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.02, 0.92, "Heart Sound Classification", fontsize=18, weight="bold")
    ax.text(0.02, 0.84, "Upload heartbeat audio (WAV/MP3)", fontsize=13)

    # Dropzone box
    ax.add_patch(plt.Rectangle((0.02, 0.18), 0.96, 0.55, fill=False, lw=2, ls="--", ec="#95a5a6"))
    ax.text(0.5, 0.45, "Drag & drop a file here\nor click to choose (mobile mic supported)",
            ha="center", va="center", fontsize=12, color="#34495e")
    ax.text(0.02, 0.1, "Tip: Record 10–20s in a quiet room for best results.", fontsize=10, color="#7f8c8d")
    savefig(fig, "screenshot-upload.png")

def result_placeholder():
    # Fake data
    sr = 4000
    t = np.linspace(0, 5, 5*sr, endpoint=False)
    y = 0.2*np.sin(2*np.pi*2*t) + 0.05*np.random.randn(len(t))
    mel = np.random.rand(64, 160) * 60 - 60
    probs = {"abnormal_other": 0.12, "normal": 0.88}

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1,1.2], width_ratios=[1,1,1], hspace=0.35, wspace=0.25)

    # Probabilities
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(list(probs.keys()), list(probs.values()), color=["#1f77b4", "#d62728"])
    ax0.set_ylim(0, 1)
    ax0.set_title("Class probabilities")
    ax0.set_ylabel("Prob")

    # Waveform
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(t, y, lw=0.7, color="#1f77b4")
    ax1.set_title("Waveform (5 s segment)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amp")

    # Spectrogram
    ax2 = fig.add_subplot(gs[1, :])
    im = ax2.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax2.set_title("Log‑Mel Spectrogram (placeholder)")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Mel bins")
    fig.colorbar(im, ax=ax2, fraction=0.015)
    savefig(fig, "screenshot-result.png")

def table_placeholder(name, title, rows=6):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.text(0.02, 0.92, title, fontsize=16, weight="bold")
    colnames = ["#", "Filename", "Primary", "Confidence"]
    data = []
    for i in range(rows):
        fn = f"rec_{i:04d}.wav"
        primary = "normal" if i % 2 == 0 else "abnormal_other"
        conf = np.random.uniform(0.7, 0.99)
        data.append([i+1, fn, primary, f"{conf*100:.2f}%"])
    # Simple table
    the_table = ax.table(cellText=data, colLabels=colnames, loc="center", cellLoc="left")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.4)
    savefig(fig, name)

def main():
    upload_placeholder()
    result_placeholder()
    table_placeholder("screenshot-batch.png", "Batch results (placeholder)")
    table_placeholder("screenshot-history.png", "History (placeholder)")

if __name__ == "__main__":
    main()