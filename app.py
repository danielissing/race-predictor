import logging
import os
import base64
import streamlit as st
import pandas as pd
import hashlib

# Enable prediction debug logging via environment variable
if os.environ.get("RACE_PREDICTOR_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("prediction").setLevel(logging.DEBUG)

# Local imports
from utils.strava import build_auth_url, exchange_code_for_token, ensure_token, list_activities, set_token_store
from utils.persistence import load_saved_app_creds, save_app_creds, forget_app_creds, load_pace_model_from_disk, save_pace_model_to_disk, load_excluded_race_ids
from utils.pace_builder import build_pace_curves_from_races
from utils.display import (
    display_course_details, display_segments_overview,
    display_prediction_results, display_pace_model_races,
    display_model_metadata, format_seconds, display_pace_curve_analysis
)
from utils.prediction import run_prediction_simulation
from models import Course, PaceModel
import config

# -- Cloud token store (session_state-backed) --
if config.is_cloud():
    _TOKEN_KEY = "_strava_tokens"
    def _cloud_save(tokens):
        st.session_state[_TOKEN_KEY] = tokens
    def _cloud_load():
        return st.session_state.get(_TOKEN_KEY)
    def _cloud_disconnect():
        st.session_state.pop(_TOKEN_KEY, None)
    set_token_store(_cloud_save, _cloud_load, _cloud_disconnect)

# -- Strava branding helpers --

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "strava")


def _load_asset_b64(filename: str) -> str:
    with open(os.path.join(_ASSETS_DIR, filename), "rb") as f:
        return base64.b64encode(f.read()).decode()


def strava_connect_button(auth_url: str):
    """Render the official 'Connect with Strava' button (Strava brand guidelines)."""
    b64 = _load_asset_b64("btn_strava_connect_with_orange_x2.png")
    st.markdown(
        f'<a href="{auth_url}" target="_blank">'
        f'<img src="data:image/png;base64,{b64}" alt="Connect with Strava" height="48"/>'
        f'</a>',
        unsafe_allow_html=True,
    )


def powered_by_strava():
    """Render the official 'Powered by Strava' attribution logo."""
    b64 = _load_asset_b64("api_logo_pwrdBy_strava_horiz_orange.png")
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" alt="Powered by Strava" height="24"/>',
        unsafe_allow_html=True,
    )


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
    creds = config.get_app_credentials()
    client_id = creds.get("client_id", "")
    client_secret = creds.get("client_secret", "")
    worker_url = config.get_worker_url()

    # With Worker, client_secret is not needed (Worker holds it)
    if client_id and (client_secret or worker_url):
        try:
            exchange_code_for_token(client_id, client_secret, code)
            if not config.is_cloud():
                save_app_creds(client_id, client_secret, config.DATA_DIR, config.APP_CREDS_PATH)
            st.success("Strava connected ✅")
            st.query_params.clear()
        except Exception as e:
            st.error("OAuth error — please try connecting again.")


def run_predictions_ui(course: Course, conditions: int):
    """
    Run predictions and update UI with results.
    Now uses single conditions parameter and clearer labels.
    """
    if st.button("Run Prediction", disabled=(not course or not st.session_state.pace_model)):
        pace_model = st.session_state.pace_model

        # Run the simplified prediction logic
        results = run_prediction_simulation(course, pace_model, conditions)

        # Store metadata
        st.session_state.prediction_meta = results["metadata"]

        # Display key info about the prediction
        meta = results["metadata"]
        altitude_slowdown = (meta['alt_speed_factor'] - 1) * 100
        rest_source = meta.get("rest_source", "fallback")
        rest_label = "learned" if rest_source == "learned" else "default"

        # Show metadata
        st.caption(
            f"🏃 Riegel exponent: `{meta['riegel_k']:.2f}` | "
            f"🅾️ Median altitude: ~`{meta['course_median_alt_m']:.0f} m` |"
            f"🏔️ Altitude slowdown: `{altitude_slowdown:.0f}%` | "
            f"⏱️ Fatigue multiplier: `x{meta.get('slow_factor_finish', 1):.2f}` | "
            f"🛑 Rest/aid time: `{format_seconds(meta.get('rest_added_finish_s', 0))}h` ({rest_label})"
        )

        # Build results dataframe with arrival/departure/rest columns
        names = [f"AS{i + 1}" for i in range(len(course.leg_end_km) - 1)] + ["Finish"]
        n_cp = len(names)
        running_only = results.get("running_only_p50", results["p50"])
        p50 = results["p50"]

        # Compute arrival, departure, rest per checkpoint
        arrivals = []
        departures = []
        rests = []
        for i in range(n_cp):
            # Rest allocated up to this checkpoint
            cumulative_rest = p50[i] - running_only[i]
            # Rest allocated up to previous checkpoint
            prev_rest = (p50[i - 1] - running_only[i - 1]) if i > 0 else 0.0
            rest_here = cumulative_rest - prev_rest

            arrival = running_only[i] + prev_rest  # arrive = running + prior rest
            departure = arrival + rest_here         # depart = arrive + rest at this stop

            is_finish = (i == n_cp - 1)
            if is_finish:
                # Finish: arrival is the final time, no departure rest
                arrival = p50[i]
                rest_here = 0.0
                departure = arrival

            arrivals.append(arrival)
            departures.append(departure)
            rests.append(rest_here)

        # Format rest as mm:ss for readability
        def fmt_rest(s):
            s = max(0, int(s))
            m, sec = divmod(s, 60)
            return f"{m}:{sec:02d}" if s > 0 else "-"

        st.session_state.eta_results = pd.DataFrame({
            "Checkpoint": names,
            "Distance (km)": [round(x, 1) for x in course.leg_end_km],
            "Arrival (P50)": [format_seconds(x) for x in arrivals],
            "Departure (P50)": [format_seconds(x) for x in departures],
            "Rest": [fmt_rest(x) for x in rests],
            "Optimistic (P10)": [format_seconds(x) for x in results["p10"]],
            "Pessimistic (P90)": [format_seconds(x) for x in results["p90"]],
        })

        # Add helpful explanation
        st.info(
            "**How to read predictions:**\n"
            "- **Arrival**: When you reach the checkpoint (running + prior rest)\n"
            "- **Departure**: When you leave (arrival + rest at this stop)\n"
            "- **Rest**: Predicted time spent at this checkpoint\n"
            "- **Optimistic (P10)**: 10% chance of being this fast\n"
            "- **Pessimistic (P90)**: 90% chance of being faster than this"
        )


# --- Main App ---
st.set_page_config(page_title="Race Time Predictor", layout="wide")
st.title("🏃‍♂️ Race Time Predictor")

# Initialize session state
if 'pace_model' not in st.session_state:
    st.session_state.pace_model = None if config.is_cloud() else load_pace_model_from_disk()
if 'course' not in st.session_state:
    st.session_state.course = None
if 'eta_results' not in st.session_state:
    st.session_state.eta_results = None
if 'excluded_race_ids' not in st.session_state:
    st.session_state.excluded_race_ids = set() if config.is_cloud() else load_excluded_race_ids()

# Handle OAuth callback
handle_oauth_callback()

# Create tabs
tab_race, tab_data = st.tabs(["🏁 Upcoming race", "📚 My data"])

# --- Sidebar ---
with st.sidebar:
    st.header("1. Strava Connection")

    if config.is_cloud():
        # Cloud mode: shared app credentials from st.secrets
        creds = config.get_app_credentials()
        client_id = creds.get("client_id", "")
        client_secret = creds.get("client_secret", "")
        redirect_uri = config.get_redirect_uri()
        worker_url = config.get_worker_url()
        callback_state = config.get_callback_state() if worker_url else ""

        tokens = ensure_token(client_id, client_secret)

        if tokens:
            st.success("Connected ✅")
            if st.button("Disconnect Strava"):
                from utils.strava import _disconnect_strava
                _disconnect_strava()
                st.rerun()
        elif client_id and (client_secret or worker_url):
            strava_connect_button(build_auth_url(client_id, redirect_uri, state=callback_state))
        else:
            st.error("Strava app credentials not configured in secrets.")
    else:
        # Local mode: user enters their own credentials
        saved = load_saved_app_creds(config.APP_CREDS_PATH)
        client_id = st.text_input("Client ID", value=saved.get("client_id", ""))
        client_secret = st.text_input("Client Secret", type="password", value=saved.get("client_secret", ""))
        redirect_uri = config.get_redirect_uri()

        tokens = ensure_token(client_id, client_secret)

        col1, col2, col3 = st.columns(3)
        with col1:
            if tokens:
                st.success("Connected ✅")
            elif client_id and client_secret:
                strava_connect_button(build_auth_url(client_id, redirect_uri))

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
                max_activities=config.MAX_ACTIVITIES, recency_mode=recency_mode,
                excluded_ids=st.session_state.excluded_race_ids,
            )
            st.session_state.pace_model = PaceModel(pace_df, used_df, meta)
            if not config.is_cloud():
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

    st.header("4. Race Conditions")

    # Single conditions slider replacing heat + feel
    conditions = st.slider(
        "Overall race conditions",
        min_value=-2,
        max_value=2,
        value=0,
        step=1,
        help="""
        Adjust based on all factors:
        • Training quality
        • Weather (heat/cold/wind/rain)
        • How you feel
        • Course technicality
        • Race importance/motivation
        """
    )

    # Show interpretation of selected value
    condition_labels = {
        -2: "😰 Terrible - Poor training, extreme weather, feeling awful",
        -1: "😕 Poor - Some issues with prep/weather/feeling",
        0: "😐 Normal - Typical race day conditions",
        1: "😊 Good - Well prepared, favorable conditions",
        2: "🚀 Perfect - Everything ideal, PR attempt!"
    }
    st.caption(condition_labels[conditions])

    st.divider()
    powered_by_strava()

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
        run_predictions_ui(course, conditions) 
        display_prediction_results()

# --- Data Tab ---
with tab_data:
    pace_model = st.session_state.pace_model
    if not pace_model:
        st.info("Build a pace model from the sidebar to see your data.")
    else:
        display_pace_curve_analysis(pace_model, course)
        display_model_metadata(pace_model)
        display_pace_model_races(pace_model, excluded_ids=st.session_state.excluded_race_ids)
