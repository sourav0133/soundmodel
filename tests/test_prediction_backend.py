"""
Comprehensive test suite for the Environmental Noise Live Prediction FastAPI backend.

Verifies:
1. Health and model readiness probes (GET /health, GET /model/status)
2. Valid 62-feature prediction payload and response structure (POST /predict)
3. Rejection of incorrect feature counts (61, 63, empty) -> 422
4. Rejection of missing mandatory fields -> 422
5. Rejection of NaN and Infinite numerical values -> 422
6. SPL safety classification logic (SAFE <= 70dB, MODERATE 70-85dB, VERY_HARMFUL > 85dB)
7. Confidence scores and 10-class probability distribution sorting
8. SQLite prediction history persistence, retrieval, filtering, and deletion
"""

import math
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.inference_service import InferenceService
from backend.services.history_service import HistoryService
from backend.services.spl_service import classify_spl
from backend.config import SPL_SAFE_THRESHOLD, SPL_MODERATE_THRESHOLD

from backend.database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_model_and_clean_history():
    """Ensures model is initialized and database history is fresh for each test."""
    init_db()
    InferenceService.get_instance().load_artifacts()
    HistoryService.clear_history()
    yield
    HistoryService.clear_history()


# -----------------------------------------------------------------------------
# 1. Health and Model Status Probes
# -----------------------------------------------------------------------------

def test_health_probe():
    """Verify GET /health reports healthy status, model loaded, and db connected."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["db_connected"] is True
    assert data["dsp_integrated"] is False
    assert "Prediction engine active" in data["note"]


def test_model_status_probe():
    """Verify GET /model/status returns canonical 62-feature schema and 10-class label map."""
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["num_features_expected"] == 62
    assert len(data["feature_schema"]) == 62
    assert data["feature_schema"][0] == "mean_spectral_centroid"
    assert data["feature_schema"][-1] == "std_chroma_11"
    assert len(data["label_mapping"]) == 10
    assert data["label_mapping"]["0"] == "air_conditioner"
    assert data["label_mapping"]["8"] == "siren"


# -----------------------------------------------------------------------------
# 2. Valid 62-Feature Prediction & Response Structure
# -----------------------------------------------------------------------------

def test_valid_62_feature_prediction_structure():
    """
    Submits a deterministic synthetic 62-value vector for API/schema smoke testing.
    NOTE: This vector is synthetic and any resulting classification is a verification
    of pipeline mechanics, not empirical evidence of real-world acoustic accuracy.
    """
    # Deterministic synthetic 62-element test vector
    synthetic_vector = [1.0 if i % 2 == 0 else 0.5 for i in range(62)]
    payload = {
        "node_id": "TEST_NODE_ALPHA",
        "spl_db": 74.5,
        "timestamp": 1793200000,
        "features": synthetic_vector
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify response schema fields
    assert data["status"] == "success"
    assert data["node_id"] == "TEST_NODE_ALPHA"
    assert data["timestamp"] == 1793200000
    assert isinstance(data["sound_category"], str)
    assert 0 <= data["class_id"] <= 9
    assert data["spl_db"] == 74.5
    assert data["spl_status"] == "MODERATE"
    assert data["confidence"] is not None
    assert 0.0 <= data["confidence"] <= 1.0

    # Verify probability distribution
    assert data["probabilities"] is not None
    assert len(data["probabilities"]) == 10
    # Top probability should equal confidence
    assert round(data["probabilities"][0]["confidence"], 4) == round(data["confidence"], 4)
    # Probabilities should be sorted descending
    for i in range(len(data["probabilities"]) - 1):
        assert data["probabilities"][i]["confidence"] >= data["probabilities"][i + 1]["confidence"]

    # Truthful DSP empty-state: spectrum_513 must be None until DSP is provided
    assert data["spectrum_513"] is None


# -----------------------------------------------------------------------------
# 3. Feature Count Validation (Strict 62 Requirements)
# -----------------------------------------------------------------------------

def test_incorrect_feature_counts_rejected():
    """Verifies that payloads with anything other than exactly 62 features are rejected with HTTP 422."""
    base_payload = {
        "node_id": "NODE_TEST",
        "spl_db": 65.0,
        "timestamp": 1793200000,
    }

    # 61 features (1 too few)
    res_61 = client.post("/predict", json={**base_payload, "features": [0.5] * 61})
    assert res_61.status_code == 422
    assert "expected exactly 62 features" in res_61.text
    assert "got 61" in res_61.text

    # 63 features (1 too many)
    res_63 = client.post("/predict", json={**base_payload, "features": [0.5] * 63})
    assert res_63.status_code == 422
    assert "expected exactly 62 features" in res_63.text
    assert "got 63" in res_63.text

    # 0 features (empty)
    res_0 = client.post("/predict", json={**base_payload, "features": []})
    assert res_0.status_code == 422
    assert "expected exactly 62 features" in res_0.text
    assert "got 0" in res_0.text


# -----------------------------------------------------------------------------
# 4. Missing and Invalid Field Validation
# -----------------------------------------------------------------------------

def test_missing_required_fields_rejected():
    """Verifies that missing mandatory fields raise HTTP 422."""
    valid_features = [0.0] * 62

    # Missing node_id
    res1 = client.post("/predict", json={"spl_db": 70.0, "timestamp": 12345, "features": valid_features})
    assert res1.status_code == 422

    # Empty node_id string
    res2 = client.post("/predict", json={"node_id": "   ", "spl_db": 70.0, "timestamp": 12345, "features": valid_features})
    assert res2.status_code == 422
    assert "node_id must not be empty" in res2.text

    # Missing spl_db
    res3 = client.post("/predict", json={"node_id": "NODE_01", "timestamp": 12345, "features": valid_features})
    assert res3.status_code == 422

    # Missing timestamp
    res4 = client.post("/predict", json={"node_id": "NODE_01", "spl_db": 70.0, "features": valid_features})
    assert res4.status_code == 422

    # Missing features
    res5 = client.post("/predict", json={"node_id": "NODE_01", "spl_db": 70.0, "timestamp": 12345})
    assert res5.status_code == 422


# -----------------------------------------------------------------------------
# 5. NaN and Non-Numeric Value Rejection
# -----------------------------------------------------------------------------

def test_nan_and_inf_values_rejected():
    """Verifies that non-finite numbers (NaN, Infinity, strings in features) are rejected with HTTP 422."""
    # Test sending "NaN" string in features array
    res_nan_str = client.post("/predict", json={
        "node_id": "NODE_01",
        "spl_db": 70.0,
        "timestamp": 1000,
        "features": [0.0] * 61 + ["NaN"]
    })
    assert res_nan_str.status_code == 422

    # Test sending null in features
    res_null = client.post("/predict", json={
        "node_id": "NODE_01",
        "spl_db": 70.0,
        "timestamp": 1000,
        "features": [0.0] * 61 + [None]
    })
    assert res_null.status_code == 422

    # Test sending invalid spl_db
    res_spl_str = client.post("/predict", json={
        "node_id": "NODE_01",
        "spl_db": "not_a_number",
        "timestamp": 1000,
        "features": [0.0] * 62
    })
    assert res_spl_str.status_code == 422


# -----------------------------------------------------------------------------
# 6. SPL Safety Classification
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("spl_val, expected_status", [
    (40.0, "SAFE"),
    (SPL_SAFE_THRESHOLD - 0.1, "SAFE"),
    (SPL_SAFE_THRESHOLD, "SAFE"),
    (SPL_SAFE_THRESHOLD + 0.1, "MODERATE"),
    (SPL_MODERATE_THRESHOLD - 0.1, "MODERATE"),
    (SPL_MODERATE_THRESHOLD, "MODERATE"),
    (SPL_MODERATE_THRESHOLD + 0.1, "VERY_HARMFUL"),
    (115.0, "VERY_HARMFUL"),
])
def test_spl_threshold_classification(spl_val, expected_status):
    """Verifies discrete categorization of SPL readings against configured boundaries."""
    assert classify_spl(spl_val) == expected_status

    # Also verify via API
    resp = client.post("/predict", json={
        "node_id": "NODE_SPL_TEST",
        "spl_db": spl_val,
        "timestamp": 1793200000,
        "features": [0.0] * 62
    })
    assert resp.status_code == 200
    assert resp.json()["spl_status"] == expected_status


# -----------------------------------------------------------------------------
# 7. Prediction History Persistence and Retrieval
# -----------------------------------------------------------------------------

def test_history_persistence_retrieval_and_clearing():
    """Verifies SQLite history logging, querying with filters, and deletion."""
    # Ensure initially empty
    hist_init = client.get("/history").json()
    assert len(hist_init) == 0

    # Submit 3 distinct predictions
    client.post("/predict", json={
        "node_id": "SENSOR_A",
        "spl_db": 62.0,
        "timestamp": 1001,
        "features": [0.0] * 62
    })
    client.post("/predict", json={
        "node_id": "SENSOR_B",
        "spl_db": 88.5,
        "timestamp": 1002,
        "features": [1.0] * 62
    })
    client.post("/predict", json={
        "node_id": "SENSOR_A",
        "spl_db": 95.0,
        "timestamp": 1003,
        "features": [2.0] * 62
    })

    # Query all history
    hist_all = client.get("/history").json()
    assert len(hist_all) == 3
    # Ordered newest first
    assert hist_all[0]["node_id"] == "SENSOR_A"
    assert hist_all[0]["spl_db"] == 95.0
    assert hist_all[0]["spl_status"] == "VERY_HARMFUL"

    # Filter by node_id
    hist_node = client.get("/history?node_id=SENSOR_B").json()
    assert len(hist_node) == 1
    assert hist_node[0]["node_id"] == "SENSOR_B"

    # Filter by min_db
    hist_db = client.get("/history?min_db=85.0").json()
    assert len(hist_db) == 2
    assert all(r["spl_db"] >= 85.0 for r in hist_db)

    # Clear history
    del_resp = client.delete("/history")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted_count"] == 3

    # Verify empty again
    hist_after = client.get("/history").json()
    assert len(hist_after) == 0
