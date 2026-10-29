"""
API Settings & Backend Health Inspector:
- Health check liveness probe (/health)
- Backend URL configuration (default http://127.0.0.1:8080)
- Model status inspection (/model/status)
- Clear History capability (/history)
- Interactive API Contract Reference
"""

import streamlit as st
import time
from datetime import datetime
from components.ui_cards import clean_html


def render_api_settings_view(api_client):
    st.html(clean_html("""
<div style="margin-bottom: 1.25rem;">
<div style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">Backend Engine & API Settings</div>
<div style="font-size: 0.82rem; color: #8B949E; margin-top: 0.2rem;">
Inspect connection health, model status, and configure prediction API endpoints.
</div>
</div>
"""))

    col1, col2 = st.columns([1.1, 1.0], gap="large")

    with col1:
        st.html(clean_html("""
<div class="storage-card">
<div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 0.5rem;">
Live Backend Connection Settings
</div>
</div>
"""))

        backend_url = st.text_input(
            "Prediction API Base URL",
            value=st.session_state.get("backend_url", "http://127.0.0.1:8080"),
            help="Target URL of the running FastAPI live prediction backend."
        )
        st.session_state.backend_url = backend_url
        api_client.base_url = backend_url.rstrip("/")

        force_mock = st.checkbox(
            "Force Standalone Mock Mode",
            value=st.session_state.get("force_mock", False),
            help="Bypasses backend HTTP calls and uses synthetic generators."
        )
        st.session_state.force_mock = force_mock
        api_client.force_mock = force_mock

        probe_cols = st.columns(2)
        with probe_cols[0]:
            if st.button("⚡ Test Health Probe (/health)", type="primary", use_container_width=True):
                health_data = api_client.check_health()
                st.session_state.health_data = health_data
                if health_data.get("status") in ["healthy", "online"]:
                    st.success(f"Backend Responded! Latency: {health_data.get('latency_ms', 0)} ms")
                else:
                    st.warning("Backend returned degraded or offline status.")

        with probe_cols[1]:
            if st.button("🗑️ Clear Test History", use_container_width=True):
                res = api_client.clear_history()
                st.toast(res.get("message", "History cleared"), icon="🧹")

        health_info = st.session_state.get("health_data", api_client.check_health())
        is_mock = health_info.get("is_mock", True)
        badge_cls = "badge-green" if not is_mock and health_info.get("status") == "healthy" else "badge-amber"

        # Model inspection details
        model_status = api_client.get_model_status()
        model_ok = model_status.get("model_loaded", False)
        model_name = model_status.get("model_name", "Unknown")
        model_badge = "badge-green" if model_ok else "badge-red"

        st.html(clean_html(f"""
<div style="margin-top: 1rem; padding: 0.85rem; background: #1B1C23; border: 1px solid #23252E; border-radius: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="font-size: 0.82rem; font-weight: 600; color: #E6EDF3;">Server Liveness</span>
<span class="badge-pill {badge_cls}">{health_info.get('status', 'offline').upper()}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="font-size: 0.82rem; font-weight: 600; color: #E6EDF3;">Production ML Model</span>
<span class="badge-pill {model_badge}">{"LOADED" if model_ok else "UNAVAILABLE"}</span>
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #8B949E; margin-top: 0.5rem; line-height: 1.6;">
Service: <span style="color: #E6EDF3;">{health_info.get('service', 'Prediction API')}</span><br/>
Model: <span style="color: #38BDF8;">{model_name} (62-D Canonical Schema)</span><br/>
Round-trip Latency: <span style="color: #F59E0B;">{health_info.get('latency_ms', 0.0)} ms</span><br/>
API Target: <span style="color: #10B981;">{api_client.base_url}</span>
</div>
</div>
"""))

    with col2:
        st.html(clean_html("""
<div class="storage-card">
<div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 0.5rem;">
API Endpoint Matrix
</div>
<div style="font-size: 0.76rem; color: #8B949E; margin-bottom: 0.75rem;">
Endpoints implemented in the live FastAPI prediction backend:
</div>

<table class="history-table">
<thead>
<tr>
<th>METHOD</th>
<th>ENDPOINT</th>
<th>PURPOSE</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="badge-pill badge-green">POST</span></td>
<td><span class="mono-path">/predict</span></td>
<td style="font-size: 0.74rem; color: #8B949E;">Live 62-D acoustic telemetry classification</td>
</tr>
<tr>
<td><span class="badge-pill badge-blue">GET</span></td>
<td><span class="mono-path">/model/status</span></td>
<td style="font-size: 0.74rem; color: #8B949E;">Model metadata, schema & label mapping</td>
</tr>
<tr>
<td><span class="badge-pill badge-blue">GET</span></td>
<td><span class="mono-path">/history</span></td>
<td style="font-size: 0.74rem; color: #8B949E;">Query persisted SQLite classification logs</td>
</tr>
<tr>
<td><span class="badge-pill badge-red">DELETE</span></td>
<td><span class="mono-path">/history</span></td>
<td style="font-size: 0.74rem; color: #8B949E;">Clear test history records</td>
</tr>
<tr>
<td><span class="badge-pill badge-blue">GET</span></td>
<td><span class="mono-path">/health</span></td>
<td style="font-size: 0.74rem; color: #8B949E;">Engine & DB readiness probe</td>
</tr>
<tr>
<td><span class="badge-pill badge-purple">DOCS</span></td>
<td><span class="mono-path">/docs</span></td>
<td style="font-size: 0.74rem; color: #8B949E;">Interactive OpenAPI / Swagger UI</td>
</tr>
</tbody>
</table>
</div>
"""))
