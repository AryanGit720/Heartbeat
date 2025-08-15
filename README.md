---
title: Heart Sound Classifier
emoji: "💓"
colorFrom: "yellow"
colorTo: "blue"
sdk: "docker"
pinned: false
license: "mit"
short_description: "Classify heart sound (phonocardiogram) recordings — normal vs abnormal"
---

# Heart Sound Classification (Heartbeat Anomaly Detection)

End‑to‑end system that detects cardiac abnormalities from heart sound recordings. It trains a CNN on log‑mel spectrograms and serves predictions via a FastAPI web app with waveform/spectrogram visuals, class probabilities, batch analysis, PDF reports, and session history.

![🤗 Spaces-Live-blue](https://github.com/user-attachments/assets/d341fcf6-8abf-4940-a2d8-80551938bdf1)<svg xmlns="http://www.w3.org/2000/svg" width="96" height="20" role="img" aria-label="🤗 Spaces: Live"><title>🤗 Spaces: Live</title><linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="r"><rect width="96" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="63" height="20" fill="#555"/><rect x="63" width="33" height="20" fill="#007ec6"/><rect width="96" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110"><text aria-hidden="true" x="325" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="530">🤗 Spaces</text><text x="325" y="140" transform="scale(.1)" fill="#fff" textLength="530">🤗 Spaces</text><text aria-hidden="true" x="785" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="230">Live</text><text x="785" y="140" transform="scale(.1)" fill="#fff" textLength="230">Live</text></g></svg>

- Live demo (Hugging Face Spaces): https://aryanbaliyan-heart-sound-classifier.hf.space  
- Code: https://github.com/AryanGit720/Heartbeat

<p align="center">
  <img src="docs/screenshot-result.png" alt="App screenshot (result view)" width="85%"/>
</p>

## Highlights
- 94% record‑level accuracy and 0.986 ROC‑AUC on PhysioNet CinC 2016 test set
- Audio pipeline: 4 kHz resampling, 5 s windows (50% overlap), 64‑mel spectrograms, global normalization
- Training: TensorFlow/Keras, SpecAugment, class weighting, EarlyStopping/ModelCheckpoint
- Web app: drag‑and‑drop uploads (mobile mic capture), waveform/spectrogram, probability bar chart, segment highlights
- Extra features: batch processing (multiple files/ZIP), PDF export, session history with delete
- Productionized: Docker image, health/readiness endpoints, deployed on Hugging Face Spaces (CPU)

## Tech Stack
- Core: Python 3.11, TensorFlow/Keras, scikit‑learn  
- Audio: librosa, soundfile, audioread/ffmpeg  
- API/UI: FastAPI, Uvicorn, Jinja2, Chart.js, Matplotlib, ReportLab  
- Data/Storage: SQLAlchemy + SQLite (session history)  
- Ops: Docker, Hugging Face Spaces, Git LFS (for model)

## Results
- Dataset: PhysioNet/CinC Challenge 2016 (normal vs abnormal), >3,000 recordings  
- Segment‑level: ~94.6% accuracy on test  
- Record‑level (avg probs across segments): ~94.2% accuracy, ROC‑AUC ~0.986

Note: Model predicts two classes: `normal`, `abnormal_other`.

---

## Screenshots

<p align="center">
  <img src="docs/screenshot-upload.png" width="85%" alt="Upload page placeholder"/>
</p>

<p align="center">
  <img src="docs/screenshot-result.png" width="85%" alt="Result page placeholder"/>
</p>

<p align="center">
  <img src="docs/screenshot-batch.png" width="85%" alt="Batch results placeholder"/>
</p>

<p align="center">
  <img src="docs/screenshot-history.png" width="85%" alt="History placeholder"/>
</p>

---


## Quickstart (local)
- Python 3.11
- ffmpeg on PATH (for MP3/M4A decoding)
- Git LFS (to fetch the model): https://git-lfs.com

### setup
git clone https://github.com/AryanGit720/Heartbeat
cd Heartbeat
git lfs install
git lfs pull  # fetches models/tf_heart_sound/best.keras

python -m venv .venv
### Windows
.venv\Scripts\activate
### macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

### Run the app
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
### Visit: http://127.0.0.1:8000

## Notes & Disclaimer
- Supported formats: WAV, MP3, M4A/AAC, OGG/WEBM (ffmpeg required). On mobile, you can record directly with your mic.
- Privacy: app stores only filenames, probabilities, and temporary files under /tmp (cleared on restart). No PHI.
- Medical disclaimer: for education/research only; not a medical device; not for diagnosis or treatment.

## Acknowledgements

- Dataset: PhysioNet/CinC Challenge 2016 — https://physionet.org/content/challenge-2016/1.0.0/
- Libraries: TensorFlow/Keras, librosa, FastAPI, Uvicorn, SQLAlchemy, Matplotlib, ReportLab
- Deployment: Hugging Face Spaces

## Maintainer

### Aryan Singh
- Demo: https://aryanbaliyan-heart-sound-classifier.hf.space
- GitHub: https://github.com/AryanGit720/Heartbeat
- Email: aryan09cc@gmail.com
- LinkedIn: https://www.linkedin.com/in/aryan-singh-b343b531a




