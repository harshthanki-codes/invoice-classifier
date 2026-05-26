"""
Test suite for the Invoice Expense Classifier.

Covers: preprocessing, model prediction accuracy, API contract, edge cases, and error handling.
Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.classifier import InvoiceClassifier, preprocess, train


# ── Preprocessing ────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_lowercases_input(self):
        assert preprocess("AWS MONTHLY BILL") == "aws monthly bill"

    def test_strips_whitespace(self):
        assert preprocess("  invoice text  ") == "invoice text"

    def test_preserves_special_chars(self):
        # Vendor names with slashes and hyphens are meaningful
        result = preprocess("Cloud/Software charges")
        assert "cloud/software" in result

    def test_empty_string(self):
        assert preprocess("") == ""


# ── Model Training & Prediction ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_classifier():
    """Train once per module — expensive to repeat."""
    train(Path("data/training_data.json"))
    InvoiceClassifier.load()
    return InvoiceClassifier


class TestClassifier:
    CATEGORY_FIXTURES = [
        ("Blue Dart courier charges for warehouse delivery", "Logistics"),
        ("AWS monthly cloud hosting bill", "Cloud/Software"),
        ("HP printer cartridges and toner refill", "Office Supplies"),
        ("BESCOM electricity bill for office premises", "Utilities"),
        ("IndiGo flight tickets for sales team", "Travel"),
        ("Raw material procurement for production batch", "Inventory"),
    ]

    @pytest.mark.parametrize("text,expected_category", CATEGORY_FIXTURES)
    def test_core_categories(self, trained_classifier, text, expected_category):
        result = trained_classifier.predict(text)
        assert result["category"] == expected_category, (
            f"Expected '{expected_category}' for: '{text}'\nGot: {result}"
        )

    def test_response_structure(self, trained_classifier):
        result = trained_classifier.predict("FedEx delivery charges")
        assert "category" in result
        assert "confidence" in result
        assert "scores" in result

    def test_confidence_is_probability(self, trained_classifier):
        result = trained_classifier.predict("Google Cloud compute bill")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self, trained_classifier):
        result = trained_classifier.predict("Jio fiber internet bill")
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-4, f"Scores sum to {total}, expected ~1.0"

    def test_all_categories_in_scores(self, trained_classifier):
        expected_categories = {
            "Logistics", "Office Supplies", "Cloud/Software",
            "Utilities", "Travel", "Inventory"
        }
        result = trained_classifier.predict("sample invoice text")
        assert expected_categories == set(result["scores"].keys())

    def test_top_score_matches_prediction(self, trained_classifier):
        result = trained_classifier.predict("Slack subscription monthly")
        top_category = max(result["scores"], key=result["scores"].get)
        assert top_category == result["category"]


# ── API Contract ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(trained_classifier):
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


class TestAPI:
    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["is_model_loaded"] is True

    def test_predict_valid_request(self, client):
        resp = client.post("/api/v1/predict", json={"text": "AWS monthly cloud hosting bill"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["category"] == "Cloud/Software"
        assert "confidence" in body
        assert "scores" in body

    def test_predict_returns_correct_schema(self, client):
        resp = client.post("/api/v1/predict", json={"text": "FedEx shipping invoice"})
        body = resp.json()
        assert isinstance(body["category"], str)
        assert isinstance(body["confidence"], float)
        assert isinstance(body["scores"], dict)

    def test_predict_empty_text_rejected(self, client):
        resp = client.post("/api/v1/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_whitespace_only_rejected(self, client):
        resp = client.post("/api/v1/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_predict_missing_text_field_rejected(self, client):
        resp = client.post("/api/v1/predict", json={})
        assert resp.status_code == 422

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_mixed_language_text(self, trained_classifier):
        # Should not crash — may have low confidence but must return valid output
        result = trained_classifier.predict("मासिक बिजली बिल for office")
        assert "category" in result
        assert "confidence" in result

    def test_very_short_text(self, trained_classifier):
        result = trained_classifier.predict("AWS")
        assert result["category"] in {
            "Logistics", "Office Supplies", "Cloud/Software",
            "Utilities", "Travel", "Inventory"
        }

    def test_numeric_heavy_text(self, trained_classifier):
        result = trained_classifier.predict("Invoice #12345 amount 5000 dated 01/06/2024")
        assert "category" in result
