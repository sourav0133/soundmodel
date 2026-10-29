"""
API Client for interacting with the Environmental Noise Live Prediction Backend.
Supports all core endpoints:
- GET  /health (Readiness and model liveness probe)
- GET  /model/status (Model metadata, 62-D schema, and label mappings)
- POST /predict (Ingests preprocessed 62-D acoustic vectors and returns verified predictions)
- GET  /history (Queries persisted predictions from SQLite)
- DELETE /history (Clears historical test records)

Maintains automatic fallback for graceful UI rendering when the backend is offline.
"""

import requests
import json
import time
from datetime import datetime
from services.mock_data import (
    generate_synthetic_audio, 
    audio_to_wav_bytes, 
    get_mock_history_records,
    ACOUSTIC_CLASSES
)


class GoAcousticApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", force_mock: bool = False):
        self.base_url = base_url.rstrip("/")
        self.force_mock = force_mock
        self.session = requests.Session()

    def check_health(self) -> dict:
        """Calls GET /health to check liveness of backend."""
        if self.force_mock:
            return {
                "status": "online (mock)",
                "service": "Standalone Mock Engine",
                "version": "1.0.0",
                "latency_ms": 1.2,
                "is_mock": True,
                "model_loaded": False,
                "dsp_integrated": False
            }

        start_time = time.time()
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=2.0)
            latency = (time.time() - start_time) * 1000
            if resp.status_code == 200:
                data = resp.json()
                data["latency_ms"] = round(latency, 1)
                data["is_mock"] = False
                return data
            else:
                return {
                    "status": "degraded",
                    "code": resp.status_code,
                    "latency_ms": round(latency, 1),
                    "is_mock": True
                }
        except Exception:
            return {
                "status": "offline",
                "service": "Standalone Mock Engine (Fallback)",
                "latency_ms": 0.0,
                "is_mock": True,
                "model_loaded": False,
                "dsp_integrated": False
            }

    def get_model_status(self) -> dict:
        """Calls GET /model/status on live backend."""
        if self.force_mock:
            return {"model_loaded": False, "error": "Mock mode enabled"}
        try:
            resp = self.session.get(f"{self.base_url}/model/status", timeout=2.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            return {"model_loaded": False, "error": str(e)}
        return {"model_loaded": False, "error": "Backend unreachable"}

    def predict_json(self, payload: dict) -> dict:
        """
        Calls POST /predict on the live backend with the preprocessed 62-D JSON payload.
        """
        if self.force_mock:
            return {
                "status": "error",
                "message": "Mock mode is active. Disable mock mode in Settings to call live prediction API.",
                "is_mock": True
            }

        try:
            resp = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json()
                data["is_live"] = True
                return data
            else:
                try:
                    err_detail = resp.json().get("detail", resp.text)
                except Exception:
                    err_detail = resp.text
                return {
                    "status": "error",
                    "http_status": resp.status_code,
                    "detail": err_detail,
                    "message": f"Backend returned HTTP {resp.status_code}: {err_detail}"
                }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Connection to backend failed at {self.base_url}/predict: {exc}"
            }

    def get_history(self, limit: int = 50, category_filter: str = "All", min_db: float = 0.0) -> list[dict]:
        """Calls GET /history to query past prediction logs from SQLite backend."""
        if not self.force_mock:
            try:
                params = {"limit": limit}
                if category_filter != "All":
                    params["category"] = category_filter
                if min_db > 0:
                    params["min_db"] = min_db
                resp = self.session.get(f"{self.base_url}/history", params=params, timeout=2.5)
                if resp.status_code == 200:
                    raw_items = resp.json()
                    # Normalize backend SQLite records to match frontend display expectations
                    normalized = []
                    for item in raw_items:
                        cat = item.get("predicted_category", item.get("category", "Unknown"))
                        spl = item.get("spl_db", item.get("raw_db", 0.0))
                        spl_status = item.get("spl_status", "")
                        node = item.get("node_id", "NODE_001")
                        conf = item.get("confidence")
                        normalized.append({
                            "id": f"rec_{item.get('id', '0')}",
                            "confidence": conf if conf is not None else 0.0,
                            "category": cat,
                            "path": f"node://{node}",
                            "node_id": node,
                            "db": f"{spl:.1f} dB ({spl_status})" if spl_status else f"{spl:.1f} dB",
                            "raw_db": spl,
                            "spl_status": spl_status,
                            "time": item.get("created_at", str(item.get("timestamp", ""))),
                            "timestamp": item.get("timestamp")
                        })
                    return normalized
            except Exception:
                pass

        records = get_mock_history_records(limit)
        # Apply local filters to mock data
        if category_filter != "All":
            records = [r for r in records if category_filter in r["category"]]
        if min_db > 0:
            records = [r for r in records if r["raw_db"] >= min_db]
        return records

    def clear_history(self) -> dict:
        """Calls DELETE /history to clear database records."""
        try:
            resp = self.session.delete(f"{self.base_url}/history", timeout=2.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
        return {"status": "error", "message": "Failed to clear history"}

    def upload_audio(self, file_bytes: bytes, filename: str) -> dict:
        """
        Placeholder for raw audio upload. Since DSP source code is unavailable,
        returns truthful status indicating DSP integration is required.
        """
        return {
            "status": "dsp_pending",
            "class": "DSP Integration Required",
            "confidence": 0.0,
            "dB": 0.0,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "probabilities": [],
            "message": "Raw audio feature extraction is blocked because the C DSP module is not integrated."
        }

    def record_session(self, action: str = "start") -> dict:
        """Calls POST /record placeholder."""
        return {
            "status": "recording" if action == "start" else "stopped",
            "session_id": f"sess_{int(time.time())}",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "message": "Live microphone streaming DSP is pending integration."
        }

    def simulate_stream_packet(self, base_type: str = "Jackhammer") -> dict:
        """
        Simulates a live streaming packet received over telemetry for UI standby demo.
        """
        import random
        jitter_db = random.uniform(-2.5, 3.5)
        conf_jitter = random.uniform(-0.03, 0.02)
        
        base_levels = {
            "Jackhammer": (96.5, "Jackhammer", 0.98),
            "Siren": (89.0, "Siren", 0.97),
            "Drilling": (87.2, "Drilling", 0.93),
            "Traffic": (74.0, "Car Horn", 0.91),
            "Ambient": (55.0, "Street Music", 0.88)
        }
        
        base_db, cat_name, base_conf = base_levels.get(base_type, (80.0, "Acoustic Noise", 0.92))
        curr_db = max(35.0, min(118.0, base_db + jitter_db))
        curr_conf = max(0.60, min(0.99, base_conf + conf_jitter))
        
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-4],
            "dB": round(curr_db, 1),
            "class": cat_name,
            "confidence": round(curr_conf, 3),
            "frame_id": f"chk_{random.randint(1000, 9999)}"
        }
