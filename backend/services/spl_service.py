"""
Sound Pressure Level (SPL) Safety Classification Service.

NOTE: These classifications (SAFE, MODERATE, VERY_HARMFUL) are provisional demo
thresholds based on general acoustic awareness guidelines (e.g., OSHA workplace 85 dBA
action criterion). They are unvalidated operational defaults intended for demonstration
and alerting, not legally-binding or medically calibrated health diagnoses.
"""

from typing import Dict, Any
from backend.config import SPL_SAFE_THRESHOLD, SPL_MODERATE_THRESHOLD


def classify_spl(spl_db: float) -> str:
    """
    Classifies a transducer-provided SPL decibel reading into a discrete safety status.
    
    Args:
        spl_db: Sound pressure level in decibels (dB SPL).

    Returns:
        One of 'SAFE', 'MODERATE', or 'VERY_HARMFUL'.
    """
    if spl_db <= SPL_SAFE_THRESHOLD:
        return "SAFE"
    elif spl_db <= SPL_MODERATE_THRESHOLD:
        return "MODERATE"
    else:
        return "VERY_HARMFUL"


def get_spl_details(spl_db: float) -> Dict[str, Any]:
    """
    Returns structured metadata explaining the SPL classification.
    """
    status = classify_spl(spl_db)
    
    if status == "SAFE":
        desc = f"Normal acoustic exposure (<= {SPL_SAFE_THRESHOLD:.1f} dB SPL)."
        action = "Standard environmental conditions; no immediate protective action required."
    elif status == "MODERATE":
        desc = (
            f"Elevated acoustic level ({SPL_SAFE_THRESHOLD:.1f} dB < SPL <= "
            f"{SPL_MODERATE_THRESHOLD:.1f} dB SPL)."
        )
        action = "Increased ambient energy; prolonged continuous exposure should be monitored."
    else:
        desc = f"Hazardous noise level (> {SPL_MODERATE_THRESHOLD:.1f} dB SPL)."
        action = "Exceeds standard 85 dB action guideline; ear protection or noise mitigation advised."

    return {
        "spl_db": spl_db,
        "spl_status": status,
        "safe_limit": SPL_SAFE_THRESHOLD,
        "moderate_limit": SPL_MODERATE_THRESHOLD,
        "description": desc,
        "guidance": action,
        "is_unvalidated_demo_threshold": True
    }
