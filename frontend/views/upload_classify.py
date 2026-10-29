"""
Upload & Classify View:
- Manual 62-D Preprocessed Telemetry Injector calling POST /predict (Developer/Test mode)
- Truthful Raw Audio Playback & DSP-Pending Notice
- Real model prediction results with confidence, SPL reading, and safety status
- Real/Truthful 513-bin Frequency Spectrum Component
- Waveform & STFT visualizations
"""

import json
import streamlit as st
import librosa
import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt

from components.audio_plots import (
    plot_waveform,
    plot_stft_spectrogram,
    plot_mel_spectrogram,
    plot_frequency_spectrum_component,
    render_confidence_breakdown_html,
    render_db_gauge_html
)
from components.ui_cards import clean_html
from services.mock_data import generate_synthetic_audio, audio_to_wav_bytes
from services.benchmark_payloads import SAMPLE_BENCHMARK_PAYLOADS


def render_upload_classify_view(api_client):
    st.html(clean_html("""
<div style="margin-bottom: 1.25rem;">
<div style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">Environmental Noise Classification</div>
<div style="font-size: 0.82rem; color: #8B949E; margin-top: 0.2rem;">
Submit 62-D preprocessed acoustic telemetry to the live model via <code style="color: #F59E0B; background: #1B1C23; padding: 2px 6px; border-radius: 4px;">POST /predict</code> or inspect audio recordings.
</div>
</div>
"""))

    col_input, col_result = st.columns([1.1, 1.1], gap="large")

    # Initialize session state for prediction result
    if "current_predict_res" not in st.session_state:
        # Default initial submission with Preset A
        default_payload = SAMPLE_BENCHMARK_PAYLOADS["Acoustic Telemetry Preset A (Children Playing Profile)"]
        init_res = api_client.predict_json(default_payload)
        st.session_state.current_predict_res = init_res
        st.session_state.current_payload = default_payload
        st.session_state.last_source = "Preset A (Synthetic Baseline)"

    with col_input:
        tab_telemetry, tab_audio = st.tabs([
            "📡 62-D DSP Telemetry Injector",
            "📁 Raw Audio Upload & Inspection"
        ])

        # TAB 1: PREPROCESSED 62-D JSON INJECTION (TEST / DEV MODE)
        with tab_telemetry:
            st.html(clean_html("""
<div class="storage-card" style="margin-bottom: 0.75rem;">
<div class="volume-sublabel">MANUAL DSP TELEMETRY INJECTION (DEV / TEST MODE)</div>
<div style="font-size: 0.75rem; color: #8B949E; margin-top: 0.25rem;">
Directly submit an actual preprocessed JSON payload to <code style="color: #38BDF8;">POST /predict</code>.
<strong>Note:</strong> This is a manual test injection for model evaluation without the C DSP library.
</div>
</div>
"""))
            selected_preset_name = st.selectbox(
                "Select Preprocessed Benchmark Vector Preset:",
                list(SAMPLE_BENCHMARK_PAYLOADS.keys()),
                index=0
            )
            preset_data = SAMPLE_BENCHMARK_PAYLOADS[selected_preset_name]

            st.caption(f"ℹ️ {preset_data.get('description', '')}")

            # Editable parameters
            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                node_id_input = st.text_input("Sensor Node ID", value=preset_data["node_id"])
            with edit_col2:
                spl_db_input = st.number_input("Transducer SPL (dB)", value=float(preset_data["spl_db"]), step=0.5)

            # Option to view/edit JSON
            with st.expander("📝 View / Edit Full 62-Feature Vector"):
                features_str = st.text_area(
                    "62 Numerical Features (JSON Array)",
                    value=json.dumps(preset_data["features"]),
                    height=120
                )

            if st.button("🚀 SUBMIT TO LIVE BACKEND (POST /predict)", type="primary", use_container_width=True):
                try:
                    parsed_features = json.loads(features_str)
                    if len(parsed_features) != 62:
                        st.error(f"Feature count mismatch: expected 62, got {len(parsed_features)}.")
                    else:
                        payload = {
                            "node_id": node_id_input,
                            "spl_db": spl_db_input,
                            "timestamp": preset_data["timestamp"],
                            "features": parsed_features
                        }
                        res = api_client.predict_json(payload)
                        st.session_state.current_predict_res = res
                        st.session_state.current_payload = payload
                        st.session_state.last_source = f"Manual Injection ({node_id_input})"
                        st.toast("✅ Live prediction executed successfully!", icon="🔊")
                        st.rerun()
                except Exception as e:
                    st.error(f"Payload parsing error: {e}")

        # TAB 2: RAW AUDIO UPLOAD & PLAYBACK
        with tab_audio:
            st.html(clean_html("""
<div class="storage-card" style="margin-bottom: 0.75rem; border-left: 4px solid #F59E0B;">
<div class="volume-sublabel" style="color: #F59E0B;">DSP INTEGRATION PENDING</div>
<div style="font-size: 0.78rem; color: #E6EDF3; margin-top: 0.35rem;">
⚠️ <strong>Raw Audio Feature Extraction Unavailable:</strong> The C DSP feature-extraction engine
is not yet integrated into this workspace. Raw audio cannot be directly classified without the 62-D extractor.
You can preview waveforms and listen to audio below, but live ML inference requires the preprocessed telemetry in Tab 1.
</div>
</div>
"""))
            uploaded_file = st.file_uploader(
                "Upload Raw Audio Clip (WAV, MP3, FLAC, OGG)",
                type=["wav", "mp3", "flac", "ogg"],
                help="Uploaded file can be previewed, but requires DSP code for feature extraction."
            )

            if uploaded_file is not None:
                bytes_data = uploaded_file.read()
                try:
                    y, sr = librosa.load(io.BytesIO(bytes_data), sr=22050)
                    st.session_state.current_audio_y = y
                    st.session_state.current_audio_sr = sr
                    st.session_state.current_filename = uploaded_file.name
                except Exception as e:
                    st.error(f"Error decoding audio: {str(e)}")

            if "current_audio_y" in st.session_state:
                st.html("""<div style="margin-top: 0.5rem;"><span class="volume-sublabel">AUDIO PLAYBACK</span></div>""")
                wav_bytes = audio_to_wav_bytes(st.session_state.current_audio_y, st.session_state.current_audio_sr)
                st.audio(wav_bytes, format="audio/wav")
                st.caption(
                    f"Loaded: {st.session_state.get('current_filename', 'audio.wav')} • "
                    f"Duration: {len(st.session_state.current_audio_y)/st.session_state.current_audio_sr:.2f}s"
                )

    # RIGHT COLUMN: INFERENCE RESULTS & DECIBEL LEVEL
    with col_result:
        res = st.session_state.get("current_predict_res", {})
        is_error = res.get("status") == "error"

        if is_error:
            st.error(f"Prediction Error: {res.get('message', 'Backend error')}")
            if "detail" in res:
                st.info(f"Details: {res['detail']}")
        else:
            pred_class = res.get("sound_category", "Unknown")
            confidence = res.get("confidence")
            db_level = res.get("spl_db", 0.0)
            spl_status = res.get("spl_status", "SAFE")
            node_id = res.get("node_id", "--")
            ts = res.get("timestamp", "--")
            raw_probs = res.get("probabilities", [])

            # Live dB Gauge
            st.html(render_db_gauge_html(db_level, peak_db=db_level + 2.5))

            # Status color mapping
            badge_map = {
                "SAFE": ("badge-green", "#10B981"),
                "MODERATE": ("badge-amber", "#F59E0B"),
                "VERY_HARMFUL": ("badge-red", "#EF4444")
            }
            spl_badge_cls, spl_border_col = badge_map.get(spl_status, ("badge-amber", "#F59E0B"))

            conf_display = f"{confidence * 100:.1f}% Confidence" if confidence is not None else "Confidence N/A"

            # Primary classification result card
            st.html(clean_html(f"""
<div class="storage-card" style="margin-bottom: 1rem; border-left: 4px solid {spl_border_col};">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<span class="volume-sublabel">CLASSIFICATION RESULT (LIVE MODEL)</span>
<div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; margin-top: 0.25rem;">
{pred_class.replace('_', ' ').title()}
</div>
<div style="font-size: 0.74rem; color: #8B949E; margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace;">
Node: <span style="color: #38BDF8;">{node_id}</span> • SPL Status: <span style="color: {spl_border_col}; font-weight: 700;">{spl_status}</span>
</div>
</div>
<div style="text-align: right;">
<span class="badge-pill {spl_badge_cls}" style="font-size: 0.82rem; padding: 0.3rem 0.7rem;">
{conf_display}
</span>
<div style="font-size: 0.72rem; color: #8B949E; margin-top: 0.35rem;">
Model: LightGBM (62-D)
</div>
</div>
</div>
</div>
"""))

            # Probability breakdown if available
            if raw_probs:
                formatted_probs = [
                    {"class": p.get("category", "").replace("_", " ").title(), "confidence": p.get("confidence", 0.0)}
                    for p in raw_probs
                ]
                st.html(render_confidence_breakdown_html(formatted_probs))

    # BOTTOM SECTION: FREQUENCY GRAPH AND SIGNAL DECONSTRUCTION
    st.html(clean_html("""
<div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
<div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">Acoustic Frequency & Spectrum Diagnostics</div>
<div style="font-size: 0.76rem; color: #8B949E;">Real-time frequency telemetry spectrum and optional raw signal representations.</div>
</div>
"""))

    # Stage 6: Frequency Spectrum Chart Component
    fig_spectrum = plot_frequency_spectrum_component(
        spectrum_data=res.get("spectrum_513"),
        title="513-Bin Acoustic Frequency Spectrum"
    )
    st.pyplot(fig_spectrum, use_container_width=True)
    plt.close(fig_spectrum)

    # If raw audio is in session, show waveform
    if "current_audio_y" in st.session_state:
        y = st.session_state.current_audio_y
        sr = st.session_state.current_audio_sr

        fig_wave = plot_waveform(y, sr, title=f"Waveform Oscillogram — {st.session_state.get('current_filename', 'Loaded Audio')}")
        st.pyplot(fig_wave, use_container_width=True)
        plt.close(fig_wave)

    with st.expander("🔍 View Raw JSON API Response (POST /predict)"):
        st.json(res)
