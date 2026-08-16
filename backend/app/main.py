"""KisanMitra backend API — offline crop-disease diagnosis & advisory."""
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import advisory, database
from .model import MODEL
from .schemas import DiagnosisResponse, ReportIn, ReportOut, MapStats

app = FastAPI(
    title="KisanMitra API",
    version="1.0.0",
    description="Offline-first crop-disease diagnosis, treatment advisory and outbreak surveillance.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


database.init_db()  # ensure table exists at import (covers TestClient & workers)


@app.on_event("startup")
def _startup():
    database.init_db()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL.ready,
            "model_version": MODEL.version, "classes": len(MODEL.labels)}


@app.get("/api/diseases")
def diseases():
    return advisory.DISEASES


@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    image: UploadFile = File(...),
    lang: str = Form("en"),
    rain: bool = Form(False),
    area_acre: float = Form(1.0),
):
    if not MODEL.ready:
        raise HTTPException(503, "Model not loaded. Train and export the model first.")
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    preds = MODEL.predict(img, topk=3)
    top = preds[0]
    label = top["label"]
    conf = top["confidence"]

    entry = advisory.DISEASES[label]
    # severity scales the disease's base severity by the model's confidence
    severity = int(round(entry["base_severity"] * (0.6 + 0.4 * conf)))
    adv = advisory.build_advisory(label, severity=severity, rain=rain, lang=lang, area_acre=area_acre)

    topk = [{"label": p["label"], "name": advisory.DISEASES[p["label"]]["name"],
             "confidence": round(p["confidence"], 4)} for p in preds]

    return DiagnosisResponse(
        confidence=round(conf, 4), topk=topk, model_version=MODEL.version, **adv,
    )


@app.post("/api/advisory", response_model=DiagnosisResponse)
def advisory_only(label: str = Form(...), lang: str = Form("en"),
                  rain: bool = Form(False), severity: int = Form(None),
                  area_acre: float = Form(1.0)):
    """Get advice for a known label without an image (used for demos/testing)."""
    if label not in advisory.DISEASES:
        raise HTTPException(404, f"Unknown label: {label}")
    adv = advisory.build_advisory(label, severity=severity, rain=rain, lang=lang, area_acre=area_acre)
    return DiagnosisResponse(confidence=1.0,
        topk=[{"label": label, "name": advisory.DISEASES[label]["name"], "confidence": 1.0}],
        model_version=MODEL.version, **adv)


@app.post("/api/reports", response_model=ReportOut)
def create_report(r: ReportIn):
    return database.add_report(r.label, r.crop, r.severity, r.lat, r.lon, r.village)


@app.get("/api/reports", response_model=MapStats)
def get_reports():
    return database.stats()
