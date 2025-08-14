from __future__ import annotations
import io
import base64
from typing import List

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def _render_prob_chart(labels: List[str], values: List[float]) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    ax.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"][: len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class probabilities")
    ax.grid(axis="y", alpha=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf

def _b64_to_buf(b64: str) -> io.BytesIO:
    return io.BytesIO(base64.b64decode(b64))

def build_pdf(
    filename: str,
    primary_prediction: str,
    confidence: float,
    labels: List[str],
    values: List[float],
    waveform_b64: str,
    spectrogram_b64: str,
    disclaimer: str,
) -> io.BytesIO:
    """
    Returns a BytesIO of the PDF report.
    """
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Heart Sound Analysis Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"File: {filename}", styles["Normal"]))
    story.append(Paragraph(f"Primary prediction: <b>{primary_prediction}</b>", styles["Normal"]))
    story.append(Paragraph(f"Confidence: <b>{confidence*100:.2f}%</b>", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Probability chart
    prob_img = _render_prob_chart(labels, values)
    story.append(Image(prob_img, width=480, height=260))
    story.append(Spacer(1, 10))

    # Waveform
    story.append(Paragraph("Waveform", styles["Heading3"]))
    story.append(Image(_b64_to_buf(waveform_b64), width=480, height=180))
    story.append(Spacer(1, 8))

    # Spectrogram
    story.append(Paragraph("Spectrogram", styles["Heading3"]))
    story.append(Image(_b64_to_buf(spectrogram_b64), width=480, height=200))
    story.append(Spacer(1, 10))

    disc = styles["BodyText"]; disc.fontSize = 9
    story.append(Paragraph(f"Disclaimer: {disclaimer}", disc))

    doc.build(story)
    out.seek(0)
    return out