"""
Test suite — Invoice Expense Classifier v2.0.0

Coverage: preprocessing · ML accuracy · GST metadata · confidence thresholds ·
          API contract · schema validation · feedback loop · edge cases · error handling

Run: pytest tests/ -v --tb=short
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.classifier import (
    CONFIDENCE_THRESHOLD,
    GST_ITC_PROFILE,
    InvoiceClassifier,
    preprocess,
    train,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def trained_model():
    train(Path("data/training_data.json"))
    InvoiceClassifier.load()
    return InvoiceClassifier


@pytest.fixture(scope="session")
def client(trained_model):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_lowercases(self):
        assert preprocess("AWS MONTHLY BILL") == "aws monthly bill"

    def test_strips_whitespace(self):
        assert preprocess("  invoice text  ") == "invoice text"

    def test_preserves_vendor_names(self):
        assert "razorpay" in preprocess("Razorpay payment gateway charges")

    def test_empty_string(self):
        assert preprocess("") == ""

    def test_special_characters_preserved(self):
        result = preprocess("Cloud/Software & hosting")
        assert "cloud/software" in result


# ── Core Category Accuracy ────────────────────────────────────────────────────

CATEGORY_FIXTURES = [
    ("Blue Dart courier charges for warehouse delivery", "Logistics"),
    ("Delhivery last-mile delivery for ecommerce", "Logistics"),
    ("FedEx express shipping invoice", "Logistics"),
    ("AWS monthly cloud hosting bill", "Cloud/Software"),
    ("Google Cloud Platform compute engine invoice", "Cloud/Software"),
    ("Razorpay payment gateway monthly charges", "Cloud/Software"),
    ("Zoho CRM enterprise subscription", "Cloud/Software"),
    ("HP printer cartridges and toner refill", "Office Supplies"),
    ("A4 paper reams bulk purchase", "Office Supplies"),
    ("Stapler and stationery items purchase", "Office Supplies"),
    ("BESCOM electricity bill for office", "Utilities"),
    ("Jio fiber business internet bill", "Utilities"),
    ("Airtel broadband monthly charges", "Utilities"),
    ("IndiGo flight tickets for sales team", "Travel"),
    ("Ola Business cab airport transfers", "Travel"),
    ("OYO Rooms accommodation for outstation team", "Travel"),
    ("Raw material procurement for production batch", "Inventory"),
    ("Steel and aluminium stock replenishment", "Inventory"),
    ("Packaging material boxes and bubble wrap", "Inventory"),
]


class TestCategoryAccuracy:
    @pytest.mark.parametrize("text,expected", CATEGORY_FIXTURES)
    def test_predicts_correct_category(self, trained_model, text, expected):
        result = trained_model.predict(text)
        assert result["category"] == expected, (
            f"'{text}'\nExpected: {expected}\nGot: {result['category']} "
            f"(conf={result['confidence']})\nScores: {result['scores']}"
        )

    def test_all_19_fixtures_consistent(self, trained_model):
        correct = sum(
            1 for text, expected in CATEGORY_FIXTURES
            if trained_model.predict(text)["category"] == expected
        )
        accuracy = correct / len(CATEGORY_FIXTURES)
        assert accuracy >= 0.85, f"Accuracy {accuracy:.0%} below 85% threshold"


# ── Prediction Response Structure ─────────────────────────────────────────────

class TestPredictionStructure:
    def test_has_required_keys(self, trained_model):
        result = trained_model.predict("FedEx delivery charges")
        for key in ("category", "confidence", "review_recommended", "scores", "gst"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_is_valid_probability(self, trained_model):
        result = trained_model.predict("AWS bill")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self, trained_model):
        result = trained_model.predict("Jio fiber internet")
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-4

    def test_top_score_matches_category(self, trained_model):
        result = trained_model.predict("Slack subscription monthly")
        top = max(result["scores"], key=result["scores"].get)
        assert top == result["category"]

    def test_all_six_categories_in_scores(self, trained_model):
        expected = {"Logistics", "Office Supplies", "Cloud/Software", "Utilities", "Travel", "Inventory"}
        result = trained_model.predict("some invoice text")
        assert expected == set(result["scores"].keys())

    def test_review_recommended_type_is_bool(self, trained_model):
        result = trained_model.predict("HP toner cartridge")
        assert isinstance(result["review_recommended"], bool)


# ── Confidence Threshold ──────────────────────────────────────────────────────

class TestConfidenceThreshold:
    def test_threshold_value_is_correct(self):
        assert CONFIDENCE_THRESHOLD == 0.72

    def test_high_confidence_does_not_recommend_review(self, trained_model):
        # Clear-cut vendor invoice — should be confident
        result = trained_model.predict("AWS EC2 and S3 cloud hosting bill")
        if result["confidence"] >= CONFIDENCE_THRESHOLD:
            assert result["review_recommended"] is False

    def test_review_flag_consistent_with_confidence(self, trained_model):
        result = trained_model.predict("miscellaneous charges invoice")
        expected_flag = result["confidence"] < CONFIDENCE_THRESHOLD
        assert result["review_recommended"] == expected_flag


# ── GST Metadata ──────────────────────────────────────────────────────────────

class TestGSTMetadata:
    def test_all_categories_have_gst_profile(self):
        categories = ["Logistics", "Office Supplies", "Cloud/Software", "Utilities", "Travel", "Inventory"]
        for cat in categories:
            assert cat in GST_ITC_PROFILE, f"Missing GST profile for {cat}"

    def test_travel_is_not_itc_eligible(self):
        assert GST_ITC_PROFILE["Travel"]["itc_eligible"] is False

    def test_cloud_software_is_itc_eligible(self):
        assert GST_ITC_PROFILE["Cloud/Software"]["itc_eligible"] is True

    def test_logistics_is_itc_eligible(self):
        assert GST_ITC_PROFILE["Logistics"]["itc_eligible"] is True

    def test_utilities_is_partial(self):
        assert GST_ITC_PROFILE["Utilities"]["itc_eligible"] == "partial"

    def test_predict_returns_gst_object(self, trained_model):
        result = trained_model.predict("IndiGo flight booking")
        assert "gst" in result
        assert "itc_eligible" in result["gst"]
        assert "hsn_sac_hint" in result["gst"]
        assert "note" in result["gst"]

    def test_travel_prediction_flags_blocked_itc(self, trained_model):
        result = trained_model.predict("IndiGo flight tickets for sales team")
        if result["category"] == "Travel":
            assert result["gst"]["itc_eligible"] is False

    def test_cloud_prediction_confirms_eligible_itc(self, trained_model):
        result = trained_model.predict("AWS monthly cloud hosting bill")
        if result["category"] == "Cloud/Software":
            assert result["gst"]["itc_eligible"] is True


# ── API Contract ──────────────────────────────────────────────────────────────

class TestAPIContract:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_body_structure(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "confidence_threshold" in body
        assert body["version"] == "2.0.0"

    def test_predict_cloud_software(self, client):
        r = client.post("/api/v1/predict", json={"text": "AWS monthly cloud hosting bill"})
        assert r.status_code == 200
        assert r.json()["category"] == "Cloud/Software"

    def test_predict_logistics(self, client):
        r = client.post("/api/v1/predict", json={"text": "Blue Dart courier charges"})
        assert r.status_code == 200
        assert r.json()["category"] == "Logistics"

    def test_predict_response_has_gst_field(self, client):
        r = client.post("/api/v1/predict", json={"text": "FedEx shipping invoice"})
        assert "gst" in r.json()

    def test_predict_response_has_review_flag(self, client):
        r = client.post("/api/v1/predict", json={"text": "FedEx shipping invoice"})
        assert "review_recommended" in r.json()

    def test_evaluate_endpoint_returns_metrics(self, client):
        # Train first to ensure metrics file exists
        client.post("/api/v1/train")
        r = client.get("/api/v1/evaluate")
        assert r.status_code == 200
        body = r.json()
        assert "cv_f1_mean" in body
        assert "production_ready" in body

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200


# ── Input Validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_text_rejected(self, client):
        assert client.post("/api/v1/predict", json={"text": ""}).status_code == 422

    def test_whitespace_only_rejected(self, client):
        assert client.post("/api/v1/predict", json={"text": "   "}).status_code == 422

    def test_missing_text_field_rejected(self, client):
        assert client.post("/api/v1/predict", json={}).status_code == 422

    def test_text_too_long_rejected(self, client):
        assert client.post("/api/v1/predict", json={"text": "x" * 1001}).status_code == 422

    def test_text_too_short_rejected(self, client):
        assert client.post("/api/v1/predict", json={"text": "ab"}).status_code == 422

    def test_feedback_invalid_category_rejected(self, client):
        r = client.post("/api/v1/feedback", json={
            "text": "some invoice",
            "correct_category": "InvalidCategory"
        })
        assert r.status_code == 422

    def test_feedback_valid_submission(self, client, tmp_path):
        # Valid feedback submission returns 200
        r = client.post("/api/v1/feedback", json={
            "text": "Unique invoice for feedback test XYZ",
            "correct_category": "Logistics"
        })
        assert r.status_code == 200
        body = r.json()
        assert "recorded_text" in body
        assert body["correct_category"] == "Logistics"


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_mixed_hindi_english(self, trained_model):
        result = trained_model.predict("मासिक बिजली बिल for office building")
        assert "category" in result

    def test_very_short_known_vendor(self, trained_model):
        result = trained_model.predict("AWS bill")
        assert result["category"] in {
            "Logistics", "Office Supplies", "Cloud/Software",
            "Utilities", "Travel", "Inventory"
        }

    def test_numeric_heavy_invoice(self, trained_model):
        result = trained_model.predict("Invoice #INV-2024-00123 amount 45000 GST 18%")
        assert "category" in result

    def test_all_caps_invoice(self, trained_model):
        result = trained_model.predict("FEDEX COURIER CHARGES FOR BULK DELIVERY")
        assert result["category"] == "Logistics"

    def test_unicode_vendor_name(self, trained_model):
        result = trained_model.predict("Société Générale bank charges for wire transfer")
        assert "category" in result
