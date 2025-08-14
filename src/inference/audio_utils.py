from pathlib import Path
from typing import List, Tuple
import io
import base64

import numpy as np
import librosa
import matplotlib.pyplot as plt

# Non-interactive backend for servers
plt.switch_backend("Agg")

# Must match training
SR = 4000
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
FMIN = 20.0
FMAX = 2000.0
SEGMENT_SECONDS = 5.0
HOP_SECONDS = 2.5

def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(int(SEGMENT_SECONDS * sr), dtype=np.float32)
    y = y.astype(np.float32)
    y = y - np.mean(y)
    y = np.clip(y, -1.0, 1.0)
    return y

def segment_audio(y: np.ndarray, sr: int = SR, seg_sec: float = SEGMENT_SECONDS, hop_sec: float = HOP_SECONDS) -> List[Tuple[int,int,float,float]]:
    seg_len = int(seg_sec * sr)
    hop_len = int(hop_sec * sr)
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)), mode="constant")
    starts = list(range(0, max(1, len(y) - seg_len + 1), hop_len))
    segments = []
    for s in starts:
        e = s + seg_len
        if e > len(y):
            pad = e - len(y)
            y_seg = np.pad(y[s:], (0, pad), mode="constant")
        else:
            y_seg = y[s:e]
        segments.append((s, e, s/sr, e/sr))
    return segments

def logmel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=1.0)
    return S_db.astype(np.float32)

def fig_to_base64(fig, dpi: int = 200) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def render_waveform_base64(y: np.ndarray, sr: int = SR) -> str:
    # Larger, clearer figure
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    t = np.arange(len(y))/sr
    ax.plot(t, y, lw=1.1, color="#1f77b4")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_title("Waveform", fontsize=14, pad=10)
    ax.grid(alpha=0.15)
    ax.margins(x=0)
    return fig_to_base64(fig, dpi=200)

def render_spectrogram_base64(y: np.ndarray, sr: int = SR) -> str:
    S = logmel(y, sr)
    fig, ax = plt.subplots(figsize=(16, 5.6), constrained_layout=True)
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Log-Mel Spectrogram", fontsize=14, pad=10)
    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Mel bins", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.025)
    return fig_to_base64(fig, dpi=200)