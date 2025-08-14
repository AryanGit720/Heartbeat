from __future__ import annotations
import os
import io
import csv
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware

from src.inference.audio_utils import load_audio, render_waveform_base64, render_spectrogram_base64, SR
from src.inference.report_utils import build_pdf
from src.app.db import init_db, add_analysis, get_history, get_analysis, delete_analysis

# Load env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Runtime-writable dirs (default to /tmp on containers)
RUNTIME_DIR = Path(os.getenv("RUNTIME_DIR", "/tmp/hsc"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(RUNTIME_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Model + metadata paths (baked into image at /app/...)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/tf_heart_sound/best.keras"))
LABEL_MAP_PATH = Path(os.getenv("LABEL_MAP_PATH", "data/metadata/label_map.json"))
NORM_PATH = Path(os.getenv("FEATURE_NORM_PATH", "data/metadata/feature_norm.json"))

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
ABNORMAL_THRESHOLD = float(os.getenv("ABNORMAL_THRESHOLD", "0.6"))
SECRET_KEY = os.getenv("SECRET_KEY", "change_me")

app = FastAPI(title="Heart Sound Classification API", version="1.4.0")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Static + templates
static_dir = Path(__file__).resolve().parent / "static"
templates_dir = Path(__file__).resolve().parent / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Lazy predictor (avoid loading TF at import time)
_predictor: Optional[Any] = None

def get_predictor():
    global _predictor
    if _predictor is None:
        # Lazy import to avoid importing TensorFlow on module import
        from src.inference.predict_tf import Predictor
        _predictor = Predictor(
            MODEL_PATH, LABEL_MAP_PATH, NORM_PATH,
            abnormal_threshold=ABNORMAL_THRESHOLD
        )
    return _predictor

class PredictionResponse(BaseModel):
    primary_prediction: str
    confidence: float
    probs: Dict[str, float]
    segments: list
    highlight_segments: list

@app.on_event("startup")
def _on_startup():
    # Initialize DB in a writable directory
    init_db()

def _get_sid(request: Request) -> str:
    sid = request.session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        request.session["sid"] = sid
    return sid

def _disclaimer() -> str:
    return (
        "This tool is for educational and research purposes only and is not a medical device. "
        "Do not use it for diagnosis or treatment. Consult a qualified healthcare professional."
    )

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}

@app.get("/ready", tags=["system"])
def ready():
    # Report "warming" until the model is first loaded
    return {"status": "ready" if _predictor is not None else "warming"}

@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "max_mb": MAX_UPLOAD_MB})

@app.post("/predict", response_class=HTMLResponse, tags=["ui"])
async def predict_ui(request: Request, file: UploadFile = File(...)):
    try:
        allowed_types = {
            "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
            "audio/x-m4a", "audio/flac", "application/octet-stream"
        }
        if file.content_type not in allowed_types:
            return templates.TemplateResponse("index.html", {
                "request": request, "error": f"Unsupported file type: {file.content_type}", "max_mb": MAX_UPLOAD_MB
            })

        raw = await file.read()
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            return templates.TemplateResponse("index.html", {
                "request": request, "error": f"File is {size_mb:.1f} MB; max allowed is {MAX_UPLOAD_MB} MB.", "max_mb": MAX_UPLOAD_MB
            })

        uid = uuid.uuid4().hex
        ext = (file.filename.split(".")[-1] or "wav").lower()
        temp_path = UPLOAD_DIR / f"{uid}.{ext}"
        with open(temp_path, "wb") as f:
            f.write(raw)

        predictor = get_predictor()  # load on first use
        result = predictor.predict_file(temp_path)

        # Visuals
        y = load_audio(temp_path, sr=SR)
        wf_b64 = render_waveform_base64(y, sr=SR)
        sp_b64 = render_spectrogram_base64(y, sr=SR)

        # Bar chart data
        probs: Dict[str, float] = result["record"]["probs"]
        labels = list(probs.keys())
        values = [probs[k] for k in labels]

        # Save to history
        sid = _get_sid(request)
        analysis_id = add_analysis(sid, file.filename, str(temp_path), result)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "filename": file.filename,
            "waveform_b64": wf_b64,
            "spectrogram_b64": sp_b64,
            "primary_prediction": result["record"]["primary_prediction"],
            "confidence_pct": f"{result['record']['confidence']*100:.2f}",
            "labels": labels,
            "values": values,
            "segments": result["segments"],
            "highlight_segments": result["highlight_segments"],
            "seg_len": result["segment_seconds"]["length"],
            "seg_hop": result["segment_seconds"]["hop"],
            "disclaimer": _disclaimer(),
            "analysis_id": analysis_id,
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, "error": f"Processing failed: {e}", "max_mb": MAX_UPLOAD_MB
        })

@app.post("/api/predict", response_model=PredictionResponse, tags=["api"])
async def api_predict(file: UploadFile = File(...)):
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return JSONResponse(status_code=413, content={"detail": f"File too large ({size_mb:.1f} MB) > {MAX_UPLOAD_MB} MB"})

    uid = uuid.uuid4().hex
    ext = (file.filename.split(".")[-1] or "wav").lower()
    temp_path = UPLOAD_DIR / f"{uid}.{ext}"
    with open(temp_path, "wb") as f:
        f.write(raw)

    predictor = get_predictor()
    result = predictor.predict_file(temp_path)
    return {
        "primary_prediction": result["record"]["primary_prediction"],
        "confidence": result["record"]["confidence"],
        "probs": result["record"]["probs"],
        "segments": result["segments"],
        "highlight_segments": result["highlight_segments"],
    }

# ---------- History ----------
@app.get("/history", response_class=HTMLResponse, tags=["ui"])
def history(request: Request):
    sid = _get_sid(request)
    rows = get_history(sid)
    items = [
        {
            "id": r.id,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
            "filename": r.filename,
            "primary_prediction": r.primary_prediction,
            "confidence": r.confidence,
        }
        for r in rows
    ]
    return templates.TemplateResponse("history.html", {"request": request, "items": items})

@app.post("/history/delete/{analysis_id}", tags=["ui"])
def history_delete(request: Request, analysis_id: int):
    sid = _get_sid(request)
    ok = delete_analysis(analysis_id, session_id=sid, delete_file=True)
    if not ok:
        return RedirectResponse(url="/history?err=not_found_or_forbidden", status_code=303)
    return RedirectResponse(url="/history?deleted=1", status_code=303)

@app.get("/view/{analysis_id}", response_class=HTMLResponse, tags=["ui"])
def view_analysis(request: Request, analysis_id: int):
    row = get_analysis(analysis_id)
    if not row:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Analysis not found", "max_mb": MAX_UPLOAD_MB})

    predictor = get_predictor()
    audio_path = Path(row.stored_path)
    result = predictor.predict_file(audio_path)
    y = load_audio(audio_path, sr=SR)
    wf_b64 = render_waveform_base64(y, sr=SR)
    sp_b64 = render_spectrogram_base64(y, sr=SR)

    probs: Dict[str, float] = result["record"]["probs"]
    labels = list(probs.keys())
    values = [probs[k] for k in labels]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": row.filename,
        "waveform_b64": wf_b64,
        "spectrogram_b64": sp_b64,
        "primary_prediction": result["record"]["primary_prediction"],
        "confidence_pct": f"{result['record']['confidence']*100:.2f}",
        "labels": labels,
        "values": values,
        "segments": result["segments"],
        "highlight_segments": result["highlight_segments"],
        "seg_len": result["segment_seconds"]["length"],
        "seg_hop": result["segment_seconds"]["hop"],
        "disclaimer": _disclaimer(),
        "analysis_id": row.id,
    })

# ---------- PDF ----------
@app.get("/report/{analysis_id}", tags=["ui"])
def pdf_report(analysis_id: int):
    row = get_analysis(analysis_id)
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Analysis not found"})

    predictor = get_predictor()
    audio_path = Path(row.stored_path)
    result = predictor.predict_file(audio_path)
    y = load_audio(audio_path, sr=SR)
    wf_b64 = render_waveform_base64(y, sr=SR)
    sp_b64 = render_spectrogram_base64(y, sr=SR)

    probs: Dict[str, float] = result["record"]["probs"]
    labels = list(probs.keys())
    values = [probs[k] for k in labels]

    pdf_io = build_pdf(
        filename=row.filename,
        primary_prediction=result["record"]["primary_prediction"],
        confidence=result["record"]["confidence"],
        labels=labels, values=values,
        waveform_b64=wf_b64, spectrogram_b64=sp_b64,
        disclaimer=_disclaimer(),
    )
    headers = {"Content-Disposition": f'attachment; filename=\"{Path(row.filename).stem}_report.pdf\""}
    return StreamingResponse(pdf_io, media_type="application/pdf", headers=headers)

# ---------- Batch ----------
@app.get("/batch", response_class=HTMLResponse, tags=["ui"])
def batch_page(request: Request):
    return templates.TemplateResponse("batch.html", {"request": request})

def _save_upload(file: UploadFile, subdir: Path) -> Path:
    safe_name = Path(file.filename).name
    out = subdir / f"{uuid.uuid4().hex}_{safe_name}"
    with open(out, "wb") as f:
        f.write(file.file.read())
    return out

def _extract_zip(file: UploadFile, subdir: Path) -> List[Path]:
    data = file.file.read()
    out_paths: List[Path] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            name = Path(m.filename).name
            if not name:
                continue
            ext = name.split(".")[-1].lower()
            if ext not in {"wav", "mp3", "flac", "m4a", "ogg"}:
                continue
            target = subdir / f"{uuid.uuid4().hex}_{name}"
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out_paths.append(target)
    return out_paths

@app.post("/batch/process", response_class=HTMLResponse, tags=["ui"])
async def batch_process(request: Request, files: List[UploadFile] = File(...)):
    try:
        sid = _get_sid(request)
        batch_dir = UPLOAD_DIR / f"batch_{uuid.uuid4().hex}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Tuple[str, Path]] = []
        for f in files:
            if f.filename.lower().endswith(".zip"):
                for p in _extract_zip(f, batch_dir):
                    saved.append((Path(p).name, p))
            else:
                p = _save_upload(f, batch_dir)
                saved.append((f.filename, p))

        if not saved:
            return templates.TemplateResponse("batch.html", {"request": request, "error": "No valid audio found in upload"})

        saved = saved[:50]

        items, ids = [], []
        predictor = get_predictor()
        for fname, apath in saved:
            result = predictor.predict_file(apath)
            analysis_id = add_analysis(sid, fname, str(apath), result)
            ids.append(str(analysis_id))
            items.append({
                "id": analysis_id,
                "filename": fname,
                "primary": result["record"]["primary_prediction"],
                "confidence": result["record"]["confidence"],
            })

        csv_url = f"/batch/csv?ids={','.join(ids)}" if ids else None
        return templates.TemplateResponse("batch_result.html", {"request": request, "items": items, "csv_url": csv_url})
    except Exception as e:
        return templates.TemplateResponse("batch.html", {"request": request, "error": f"Batch failed: {e}"})

@app.get("/batch/csv", tags=["ui"])
def batch_csv(ids: str):
    from src.app.db import get_analysis as _get
    id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    rows = [_get(i) for i in id_list if _get(i)]
    def _stream():
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id", "created_at", "filename", "primary", "confidence"])
        for r in rows:
            w.writerow([r.id, r.created_at.isoformat(timespec="seconds"), r.filename, r.primary_prediction, f"{r.confidence:.6f}"])
        yield out.getvalue()
    headers = {"Content-Disposition": "attachment; filename=batch_results.csv"}
    return StreamingResponse(_stream(), media_type="text/csv", headers=headers)