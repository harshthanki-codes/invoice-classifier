"""
Invoice Expense Classifier — FastAPI application entry point.

Startup trains or loads the model once before accepting traffic.
All prediction handlers are CPU-bound and synchronous — no async
I/O, so there's no benefit to async route handlers here.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.ml.classifier import InvoiceClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Invoice Classifier API starting — loading model...")
    InvoiceClassifier.load()
    logger.info("Model ready. Serving traffic.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Invoice Expense Classifier",
    description="""
## Invoice Expense Classifier API

ML-powered REST API that classifies free-text invoice descriptions into
structured expense categories — purpose-built for GST compliance workflows.

### Features
- **6 expense categories** aligned to Indian SMB accounting
- **Calibrated confidence scores** with a human-review flag below 0.72
- **GST/ITC guidance** per prediction — maps to CGST Act Section 17(5)
- **Feedback loop** — correct predictions feed the next retrain automatically
- **Hot reload** — retrain and redeploy without restarting the server

### Categories
`Logistics` · `Office Supplies` · `Cloud/Software` · `Utilities` · `Travel` · `Inventory`
""",
    version="2.0.0",
    lifespan=lifespan,
    contact={
        "name": "API Support",
        "url": "https://github.com/your-username/invoice-classifier",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Invoice Expense Classifier",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
