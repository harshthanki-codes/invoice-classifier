<div align="center">

# 🧾 Invoice Expense Classifier

**Production-ready ML API for automated invoice expense categorisation**  
Built for GST-compliant finance workflows in Indian SMBs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen?logo=pytest)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live%20on%20Render-46E3B7?logo=render)](https://invoice-classifier-h3kj.onrender.com)

[**Live API**](https://invoice-classifier-h3kj.onrender.com/docs) · [**Docs**](#api-reference) · [**Quickstart**](#quickstart)

</div>

---

## Overview

Finance and ERP systems routinely receive unstructured invoice text from vendors. Manually routing these to the correct GL account or GST expense category is slow and error-prone. This service automates that classification with a lightweight ML model that:

- Returns predictions in **< 5ms** with zero GPU dependency
- Includes **calibrated confidence scores** and a human-review flag for low-certainty predictions
- Maps every prediction to **GST/ITC eligibility** under the CGST Act — so downstream systems can auto-populate input tax credit fields without a second lookup
- Exposes a **feedback endpoint** that lets users correct wrong predictions, feeding a continuous improvement loop

---

## Architecture

```
POST /api/v1/predict
        │
        ▼
┌─────────────────────┐
│  Pydantic Validator │  ← rejects blank, too-short, too-long text
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Vectoriser  │  ← unigrams + bigrams + trigrams, 6000 features
│  (sublinear_tf)     │    sublinear TF dampens high-frequency boilerplate
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Logistic Regression │  ← C=8, balanced class weights, lbfgs solver
│  (multinomial)      │    predict_proba → calibrated confidence scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Response                                           │
│  category · confidence · review_recommended · gst  │
└─────────────────────────────────────────────────────┘
```

**Why TF-IDF + Logistic Regression?**

For short, domain-specific text with a fixed label set, this combination outperforms Naive Bayes and matches fine-tuned transformers at a fraction of operational cost. Key advantages for a fintech context:

| Property | TF-IDF + LR | BERT/Transformer |
|---|---|---|
| Inference latency | ~1ms | 200–500ms |
| GPU required | No | Recommended |
| Explainability | Full (feature weights) | Limited |
| Cold start | ~200ms | 8–15s |
| Cost | $0 | GPU instance |

The decision reverses if the label taxonomy grows beyond ~20 categories or if semantic understanding of ambiguous text becomes critical.

---

## Model Performance

> All metrics from 5-fold stratified cross-validation on 185 labelled Indian invoice samples.

**CV F1-macro: `0.8213 ± 0.0386`**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cloud/Software | 0.84 | 0.89 | **0.86** | 35 |
| Travel | 0.89 | 0.83 | **0.86** | 30 |
| Inventory | 0.81 | 0.87 | **0.84** | 30 |
| Logistics | 0.77 | 0.90 | **0.83** | 30 |
| Office Supplies | 0.85 | 0.73 | **0.79** | 30 |
| Utilities | 0.81 | 0.73 | **0.77** | 30 |

**Confidence threshold: `0.72`** — predictions below this return `review_recommended: true`. Empirically, sub-threshold predictions have a ~3× higher error rate. In production, route these to a human review queue rather than auto-approving them.

> To reach F1 > 0.90, add 100+ diverse samples per category and retrain via `POST /api/v1/train`.

---

## GST / ITC Alignment

Every prediction includes GST treatment guidance derived from the CGST Act:

| Category | ITC Eligible | Statutory Basis |
|---|---|---|
| Logistics | ✅ Yes | SAC 9965 — forward charge mechanism |
| Cloud/Software | ✅ Yes | SAC 9983 — IT services |
| Office Supplies | ✅ Yes | HSN 4820/8443 (food/beverages blocked) |
| Inventory | ✅ Yes | HSN varies by commodity |
| Utilities | ⚠️ Partial | Electricity exempt; telecom/internet eligible |
| Travel | ❌ No | Blocked — CGST Act Section 17(5)(b) |

---

## Quickstart

### Local (Python 3.10+)

```bash
git clone https://github.com/your-username/invoice-classifier
cd invoice-classifier

pip install -r requirements.txt

# Train the model
python scripts/train.py

# Start the API
uvicorn app.main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

### Docker

```bash
docker build -t invoice-classifier .
docker run -p 8000:8000 invoice-classifier
```

### Docker Compose

```bash
docker-compose up --build
```

---

## API Reference

### `POST /api/v1/predict`

Classify an invoice description.

**Request**
```json
{ "text": "AWS monthly cloud hosting bill" }
```

**Response**
```json
{
  "category": "Cloud/Software",
  "confidence": 0.9134,
  "review_recommended": false,
  "scores": {
    "Cloud/Software": 0.9134,
    "Utilities": 0.0421,
    "Office Supplies": 0.0198,
    "Logistics": 0.0142,
    "Travel": 0.0071,
    "Inventory": 0.0034
  },
  "gst": {
    "itc_eligible": true,
    "hsn_sac_hint": "SAC 9983 — Information Technology Services",
    "note": "ITC fully available on SaaS, cloud, and software subscriptions"
  }
}
```

---

### `POST /api/v1/feedback`

Submit a correction to improve future model versions.

```json
{
  "text": "Swiggy Dineout client entertainment bill",
  "correct_category": "Travel"
}
```

Response:
```json
{
  "message": "Correction recorded. Run /train to apply.",
  "recorded_text": "Swiggy Dineout client entertainment bill",
  "correct_category": "Travel",
  "total_training_samples": 186
}
```

---

### `POST /api/v1/train`

Retrain on current data (including any feedback corrections) and hot-reload.

```json
{
  "message": "Model retrained and hot-reloaded.",
  "cv_f1_mean": 0.8213,
  "cv_f1_std": 0.0386,
  "classes": ["Cloud/Software","Inventory","Logistics","Office Supplies","Travel","Utilities"],
  "num_samples": 185,
  "confidence_threshold": 0.72,
  "production_ready": false
}
```

---

### `GET /api/v1/evaluate`

Fetch metrics from last training run without retraining.

---

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "2.0.0",
  "confidence_threshold": 0.72
}
```

---

## More API Examples

```bash
# Predict
curl -X POST https://invoice-classifier-h3kj.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Jio fiber business internet connection bill"}'

# Submit feedback
curl -X POST https://invoice-classifier-h3kj.onrender.com/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "Team dinner at client office", "correct_category": "Travel"}'

# Retrain after collecting feedback
curl -X POST https://invoice-classifier-h3kj.onrender.com/api/v1/train

# Check model health
curl https://invoice-classifier-h3kj.onrender.com/api/v1/health
```

---

## Testing

```bash
pytest tests/ -v
# 62 passed in ~8s
```

Covers: preprocessing · category accuracy (19 fixtures) · response schema · confidence thresholds · GST metadata · API contract · input validation · feedback loop · edge cases (Hindi-English mixed, all-caps, numeric-heavy, unicode).

---

## Project Structure

```
invoice-classifier/
├── app/
│   ├── api/routes.py          # /predict /feedback /train /evaluate /health
│   ├── ml/classifier.py       # TF-IDF + LR pipeline, GST profiles, feedback loop
│   ├── schemas/invoice.py     # Pydantic v2 request/response contracts
│   └── main.py                # App factory, lifespan, global error handler
├── data/
│   └── training_data.json     # 185 labelled Indian invoice samples with GST tags
├── reports/
│   └── evaluation.md          # Auto-generated per-class F1 + confusion matrix
├── models/                    # Persisted artifacts (git-ignored)
├── scripts/
│   ├── train.py               # CLI training with CV metrics
│   └── evaluate.py            # Per-class F1 + markdown report generator
├── tests/
│   └── test_classifier.py     # 62 tests across 7 test classes
├── .github/workflows/ci.yml   # GitHub Actions — test + docker smoke test
├── Dockerfile                 # Multi-stage, non-root, pre-trained at build time
├── docker-compose.yml
└── requirements.txt
```

---

## Deployment

The container pre-trains the model at image build time — startup is instant.

**Render (live)**
Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Fly.io / Railway**
Both support Dockerfile deployments directly. No additional config needed.

**Scaling note:** The model is loaded in-process. For multi-worker deployments (`--workers 4`), the model is loaded once per worker process — ~2MB per worker, negligible.

---

## Roadmap

- [ ] HSN/SAC code prediction alongside category
- [ ] Vendor entity extraction (NER) for structured output
- [ ] Confidence threshold tuning endpoint
- [ ] Webhook support for async prediction results
- [ ] Multi-language invoice support (Hindi, Tamil, Gujarati)

---

## License

MIT — use freely, contribute back.
