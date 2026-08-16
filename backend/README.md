# KisanMitra — Backend & ML

Offline-first crop-disease diagnosis, treatment advisory and outbreak surveillance
for smallholder farmers. This folder contains the **trained model**, the **training
pipeline**, and the **FastAPI backend** that serves diagnoses to the app.

Built for HackFusion @ CIT.

---

## What's inside

```
backend/
├── app/                     FastAPI service
│   ├── main.py              API endpoints
│   ├── model.py             TFLite inference wrapper (the "on-device" model)
│   ├── advisory.py          Advisory brain: treatment + economics + spoken advice
│   ├── database.py          SQLite store for outbreak reports
│   ├── schemas.py           Pydantic request/response models
│   └── data/
│       ├── diseases.json    Knowledge base: 15 classes → treatment, dose, PHI, economics
│       └── reports.db       (created at runtime)
├── models/                  Trained artifacts (produced by training)
│   ├── model.tflite         Quantized edge model  ← ships into the Flutter app
│   ├── model.keras          Full Keras model
│   ├── labels.json          Class order
│   └── metrics.json         Accuracy + per-class scores
├── train/
│   ├── gen_data.py          Procedural dataset (used here — see note below)
│   ├── prep_data.py         REAL PlantVillage loader (use in production)
│   └── train.py             Trains, evaluates, exports TFLite
├── tests/test_api.py        End-to-end API smoke test
└── requirements.txt
```

## The model

The classifier covers **15 classes** across tomato, potato and bell-pepper/chili
(bacterial spot, early/late blight, leaf mold, septoria, spider mites, target spot,
yellow leaf curl & mosaic virus, plus healthy for each crop) — the crops for which
real PlantVillage data exists.

> **Training data.** The shipped model is trained on **real PlantVillage leaf
> photos** — 7,025 images (up to 500 per class) pulled from the GitHub mirror
> `spMohanty/PlantVillage-Dataset` via `prep_real.py`. It reaches **91.4% validation
> accuracy on 1,053 held-out real photos** (`metrics.json`), and classifies all 6
> real sample images bundled in the app correctly. `gen_data.py` (a procedural-image
> generator) and `prep_data.py` (the TFDS loader) remain in the repo as alternate
> data sources; the downstream `train.py`, export and backend code are identical
> regardless of which dataset you use.

## Run the backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

## Retrain on REAL PlantVillage data (production)

```bash
python train/prep_data.py     # downloads PlantVillage, builds the subset arrays
python train/train.py         # trains, evaluates, writes models/model.tflite
```
For real photos, switch `train.py` to transfer learning (uncomment MobileNetV2,
`weights="imagenet"`) for higher accuracy — the export path is identical.

## API

| Method | Path             | Purpose                                                        |
|--------|------------------|----------------------------------------------------------------|
| GET    | `/health`        | Model status & version                                         |
| GET    | `/api/diseases`  | Full knowledge base                                            |
| POST   | `/api/diagnose`  | **Image → disease + advisory + economics + spoken advice**     |
| POST   | `/api/advisory`  | Advice for a known label (no image; for demos)                 |
| POST   | `/api/reports`   | Submit a geotagged outbreak report                             |
| GET    | `/api/reports`   | Outbreak map data + stats (reports / villages / alerts)        |

### `POST /api/diagnose`
Multipart form: `image` (file), `lang` (`en|hi|ta|te|mr`), `rain` (bool),
`area_acre` (float). Returns disease, confidence, severity, full treatment plan,
a rupee cost–benefit, a weather-aware recommendation, and a ready-to-speak
sentence in the chosen language (the app passes `spoken_advice` to on-device TTS).

## How the pieces map to the pitch

- **On-device model** → `model.tflite` (quantized, runs offline in the Flutter app).
- **Decision, not diagnosis** → `advisory.py` economic engine (rupee cost–benefit + weather timing).
- **Vernacular voice** → `spoken_advice` string per language → device TTS / Bhashini.
- **Network-effect moat** → `/api/reports` outbreak surveillance in `database.py`.
