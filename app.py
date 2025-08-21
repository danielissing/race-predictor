import streamlit as st
import pandas as pd
import hashlib

# Local imports
from utils.strava import build_auth_url, exchange_code_for_token, ensure_token, list_activities
from utils.persistence import load_saved_app_creds, save_app_creds, forget_app_creds, load_pace_model_from_disk, save_pace_model_to_disk
from utils.pace_builder import build_pace_curves_from_races
from utils.display import (
    display_course_details, display_segments_overview,
    display_prediction_results, display_pace_model_races,
    display_model_metadata, format_seconds, display_pace_curve_analysis
)
from utils.prediction import run_prediction_simulation
from models import Course, PaceModel
import config

# -- UI helper functions --

def get_course_from_session(aid_km_text: str, aid_units: str):
    """Creates and caches a Course object in the session state."""
    gpx_bytes = st.session_state.get("gpx_bytes")
    if not gpx_bytes:
        st.session_state.course = None
        return None

    fp = hashlib.md5(gpx_bytes).hexdigest() + f"|{aid_km_text}|{aid_units}"

    if st.session_state.get("course_fp") != fp:
        try:
            st.session_state.course = Course(gpx_bytes, aid_km_text, aid_units)
            st.session_state.course_fp = fp
        except Exception as e:
            st.error(f"Failed to process GPX: {e}")
            st.session_state.course = None

    return st.session_state.get("course")


def handle_oauth_callback():
    """Handle OAuth callback from Strava."""
    qs = st.query_params
    if "code" not in qs:
        return

    code = qs["code"]
    saved = load_saved_app_creds(config.APP_CREDS_PATH)
    client_id = saved.get("client_id", "")
    client_secret = saved.get("client_secret", "")

    if client_id and client_secret:
        try:
            exchange_code_for_token(client_id, client_secret, code)
            save_app_creds(client_id, client_secret, config.DATA_DIR, config.APP_CREDS_PATH)
            st.success("Strava connected ✅")
            st.query_params.clear()
        except Exception as e:
            st.error(f"OAuth error: {e}")


def run_predictions_ui(course: Course, feel: str, heat: str):
    """Run predictions and update UI with results."""
    if st.button("Run Prediction", disabled=(not course or not st.session_state.pace_model)):
        pace_model = st.session_state.pace_model

        # Run the core prediction logic
        results = run_prediction_simulation(course, pace_model, feel, heat)

        # Store metadata
        st.session_state.prediction_meta = results["metadata"]

        # Build results dataframe
        names = [f"AS{i + 1}" for i in range(len(course.leg_end_km) - 1)] + ["Finish"]
        st.session_state.eta_results = pd.DataFrame({
            "Segment": names,
            "Km": [round(x, 1) for x in course.leg_end_km],
            "Arrival time (best guess)": [format_seconds(x) for x in results["p50"]],
            "P10 (Optimistic)": [format_seconds(x) for x in results["p10"]],
            "P90 (Pessimistic)": [format_seconds(x) for x in results["p90"]],
        })


# --- Main App ---
st.set_page_config(page_title="Race Time Predictor", layout="wide")
st.title("🏃‍♂️ Race Time Predictor")

# Initialize session state
if 'pace_model' not in st.session_state:
    st.session_state.pace_model = load_pace_model_from_disk()
if 'course' not in st.session_state:
    st.session_state.course = None
if 'eta_results' not in st.session_state:
    st.session_state.eta_results = None

# Handle OAuth callback
handle_oauth_callback()

# Create tabs
tab_race, tab_data = st.tabs(["🏁 Upcoming race", "📚 My data"])

# --- Sidebar ---
with st.sidebar:
    st.header("1. Strava Connection")
    saved = load_saved_app_creds(config.APP_CREDS_PATH)
    client_id = st.text_input("Client ID", value=saved.get("client_id", ""))
    client_secret = st.text_input("Client Secret", type="password", value=saved.get("client_secret", ""))

    tokens = ensure_token(client_id, client_secret)

    col1, col2, col3 = st.columns(3)
    with col1:
        if tokens:
            st.success("Connected ✅")
        elif client_id and client_secret:
            st.link_button("Connect", url=build_auth_url(client_id, "http://localhost:8501"))

    with col2:
        if st.button("Save creds", disabled=not (client_id and client_secret)):
            save_app_creds(client_id, client_secret, config.DATA_DIR, config.APP_CREDS_PATH)
            st.success("Saved!")

    with col3:
        if st.button("Forget creds"):
            forget_app_creds(config.APP_CREDS_PATH)
            st.success("Forgotten!")
            st.rerun()

    st.header("2. Build Pace Model")
    recency_mode = st.select_slider("Recency Weighting", ["off", "mild", "medium"], value="mild")
    if st.button("Build from my Strava races", disabled=not tokens):
        with st.spinner("Fetching races and building model..."):
            acts = list_activities(tokens["access_token"])
            pace_df, used_df, meta = build_pace_curves_from_races(
                tokens["access_token"], acts, config.GRADE_BINS,
                max_activities=config.MAX_ACTIVITIES, recency_mode=recency_mode
            )
            st.session_state.pace_model = PaceModel(pace_df, used_df, meta)
            save_pace_model_to_disk(st.session_state.pace_model)
            st.success("Pace model built!")

    st.header("3. Course Details")
    gpx_file = st.file_uploader("Upload race GPX", type=["gpx"])
    if gpx_file:
        st.session_state.gpx_bytes = gpx_file.getvalue()

    aid_km_text = st.text_input("Aid stations (cumulative km)", "10, 21.1, 32, 42.2")
    aid_units = st.radio("Aid station units", ["km", "mi"], horizontal=True)
    st.caption("All outputs are in metric (km). This only affects input parsing.")

    course = get_course_from_session(aid_km_text, aid_units)

    st.header("4. Race Day Conditions")
    feel = st.selectbox("How do you feel?", ["good", "ok", "meh"], index=1)
    heat = st.selectbox("Heat", ["cool", "moderate", "hot"], index=0)

# --- Race Tab ---
with tab_race:
    if not course:
        st.info("Upload a GPX file in the sidebar to get started.")
    else:
        display_course_details(course)
        st.divider()

        st.subheader("Segments Overview")
        display_segments_overview(course)

        st.divider()
        st.subheader("Predictions")
        run_predictions_ui(course, feel, heat)
        display_prediction_results()

# --- Data Tab ---
with tab_data:
    pace_model = st.session_state.pace_model
    if not pace_model:
        st.info("Build a pace model from the sidebar to see your data.")
    else:
        display_pace_curve_analysis(pace_model, course)
        display_model_metadata(pace_model)
        display_pace_model_races(pace_model)
