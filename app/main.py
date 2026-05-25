"""
Invoice Expense Classifier — FastAPI application.

Startup loads (or trains) the ML model once. All prediction requests are stateless
and synchronous — no async DB calls, so sync handlers are fine here.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.ml.classifier import InvoiceClassifier

# Structured logging to stdout — plays well with Docker and log aggregators
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model before the server starts accepting traffic."""
    logger.info("Starting Invoice Classifier API — loading model...")
    InvoiceClassifier.load()
    logger.info("Model ready. API is live.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Invoice Expense Classifier",
    description=(
        "ML-powered API that classifies free-text invoice descriptions "
        "into structured expense categories. Built with TF-IDF + Logistic Regression."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Invoice Classifier API v1.0.0 — see /docs for usage."}
