"""
Benchmark and sample telemetry payloads for manual test injection in the Streamlit frontend.
Allows developers and users to test POST /predict on the live backend with verified 62-D vectors.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_PATH = PROJECT_ROOT / "ML_model" / "reports" / "feature_statistics.json"
SCHEMA_PATH = PROJECT_ROOT / "ML_model" / "configs" / "feature_schema.json"


def _load_schema_and_means() -> List[float]:
    """Loads feature schema and baseline mean values from feature_statistics.json."""
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        feature_names = schema.get("feature_names", [])

        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)

        return [float(stats[name]["mean"]) for name in feature_names]
    except Exception:
        # Fallback to zero vector if reports missing
        return [0.0] * 62


BASELINE_MEANS = _load_schema_and_means()

SAMPLE_BENCHMARK_PAYLOADS: Dict[str, Dict[str, Any]] = {
    "Acoustic Telemetry Preset A (Children Playing Profile)": {
        "node_id": "NODE_PARK_001",
        "spl_db": 68.4,
        "timestamp": 1793200100,
        "description": "Baseline acoustic telemetry vector derived from dataset statistical means.",
        "features": list(BASELINE_MEANS)
    },
    "Acoustic Telemetry Preset B (Emergency Siren Profile)": {
        "node_id": "NODE_METRO_042",
        "spl_db": 91.5,
        "timestamp": 1793200200,
        "description": "Harmonic spectral pattern with elevated high-frequency components.",
        "features": [1.0 if i % 2 == 0 else 0.85 for i in range(62)]
    },
    "Acoustic Telemetry Preset C (Traffic & Car Horn Profile)": {
        "node_id": "NODE_INTERSECT_07",
        "spl_db": 84.2,
        "timestamp": 1793200300,
        "description": "Impulsive high-contrast pattern typical of traffic alert signatures.",
        "features": [0.0 if i % 3 != 0 else 1.5 for i in range(62)]
    },
    "Acoustic Telemetry Preset D (Heavy Mechanical Engine Profile)": {
        "node_id": "NODE_INDUSTRIAL_12",
        "spl_db": 76.8,
        "timestamp": 1793200400,
        "description": "Low-frequency energy distribution with steady mechanical centroid.",
        "features": [50.0 + (i * 0.5) for i in range(62)]
    },
    "Acoustic Telemetry Preset E (Hazardous Noise >85 dB Test)": {
        "node_id": "NODE_QUARRY_99",
        "spl_db": 97.8,
        "timestamp": 1793200500,
        "description": "High decibel telemetry (>85 dB OSHA threshold) to verify VERY_HARMFUL alerting.",
        "features": [10.0 + (i * 0.2) for i in range(62)]
    }
}
