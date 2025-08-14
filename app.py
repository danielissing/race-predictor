import os, json
import streamlit as st
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import hashlib
from utils.gpx import parse_gpx, legs_from_aid_stations, distance_by_grade_bins, segment_stats, aid_station_markers, resample_with_grade, apply_ultra_adjustments_progressive, parse_cumulative_dist, build_pace_curves_from_races, altitude_impairment_multiplicative
from models.predictor import simulate_etas
from utils.strava_utils import build_auth_url, exchange_code_for_token, ensure_token, list_activities
from utils.app_utils import load_saved_app_creds, save_app_creds, forget_app_creds, fmt

# CONSTANTS
DATA_DIR = "data"
APP_CREDS_PATH = Path(DATA_DIR) / "strava_app.json"
GRADE_BINS = [-100, -30, -20, -12, -8, -5, -3, 0, 3, 5, 8, 12, 20, 30, 100] # used to calculate average speed by grade
MC_SIMS = 1000  # fixed Monte Carlo runs
DEFAULT_RIEGEL_K =  1.06 # coefficient to account for slowdown over longer distances. Default value is common for road running.
STEP_LENGTH = 10.0 # length of minimum segment for which we'll calculate the grade (in m)
STEP_WINDOW = 40.0 # size of sliding window (in m) used to smooth out gpx data
MAX_ACTIVITIES = 200 # upper limit for how much activities to load from strava (due to API restrictions)
CLUSTER_RADIUS = 200.0 # aid stations within this radius (in m) will be merged on the map
EPSILON = 1e-6

st.set_page_config(page_title="Race Time Predictor", layout="wide")
st.title("🏃‍♂️ Race Time Predictor")
tab_race, tab_data = st.tabs(["🏁 Upcoming race", "📚 My data"])

# load pace curves and past races if data was already saved (to reduce number of API calls)
if 'pace_df' not in st.session_state and os.path.exists("data/pace_curves.csv"):
    st.session_state['pace_df'] = pd.read_csv("data/pace_curves.csv")
if 'used_races' not in st.session_state and os.path.exists("data/used_races.csv"):
    st.session_state['used_races'] = pd.read_csv("data/used_races.csv")
pc = st.session_state.get('pace_df')

with st.sidebar:
    st.header("Strava")
    st.caption("Create a Strava API app → set Authorization Callback Domain to **localhost**.")

    # 1) Prefill from saved creds → env vars → session (so you rarely retype)
    saved = load_saved_app_creds(APP_CREDS_PATH)
    default_id = saved.get("client_id") or os.environ.get("STRAVA_CLIENT_ID", "")
    default_secret = saved.get("client_secret") or os.environ.get("STRAVA_CLIENT_SECRET", "")

    client_id = st.text_input("Client ID", value=default_id, key="client_id")
    client_secret = st.text_input("Client Secret", type="password", value=default_secret, key="client_secret")
    redirect_uri = "http://localhost:8501"  # fixed

    # 2) OAuth callback handling
    qs = st.query_params
    if "code" in qs and client_id and client_secret:
        code = qs["code"]
        try:
            exchange_code_for_token(client_id, client_secret, code)
            save_app_creds(client_id, client_secret, DATA_DIR, APP_CREDS_PATH)  # save on first successful connect
            st.success("Strava connected ✅")
            st.query_params.clear()
        except Exception as e:
            st.error(f"OAuth error: {e}")

    # 3) Try refresh with whatever we loaded; if it fails, tokens=None and we'll show Connect
    tokens = ensure_token(client_id, client_secret)

    # 4) UI: Connect / Disconnect / Save / Forget
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if tokens:
            st.success("Connected ✅")
        elif client_id and client_secret:
            st.link_button("Connect Strava", url=build_auth_url(client_id, redirect_uri), use_container_width=True)
        else:
            st.info("Enter Client ID/Secret to enable Connect.")

    with colB:
        if st.button("Save app creds", use_container_width=True, disabled=not (client_id and client_secret)):
            save_app_creds(client_id, client_secret, DATA_DIR, APP_CREDS_PATH)
            st.success("Saved locally")

    with colC:
        if st.button("Forget creds", use_container_width=True):
            forget_app_creds(APP_CREDS_PATH)
            st.success("Forgot saved creds")

    # Add button to build pace curves
    st.divider()
    st.header("Build pace curves")
    st.caption("Uses your Strava **races** only (Run Type = Race).")
    build_btn = st.button("Build from my Strava races", disabled=(tokens is None))

    # Field to upload upcoming race gpx
    st.header("Course")
    gpx_file = st.file_uploader("Upload race GPX", type=["gpx"])
    # Read the uploaded GPX once and reuse (avoid empty second read)
    if gpx_file is not None:
        try:
            st.session_state['gpx_bytes'] = gpx_file.getvalue()
        except Exception:
            # Fallback if getvalue() isn't available
            st.session_state['gpx_bytes'] = gpx_file.read()
    else:
        st.session_state.pop('gpx_bytes', None)

    # Provide distance markers (cumulative) for the aid stations
    # You can enter in either miles or km, but output will by design be in metric units
    aid_km_text = st.text_input("Aid stations (cumulative distances)", value="10, 21, 33, 50")
    aid_units = st.radio("Aid station units", options=["km", "mi"], index=0, horizontal=True)
    st.caption("All outputs are in metric (km). This only affects the input parsing.")

    # Allow user to add some simple race day conditions - form, weather, and how much weight to put on more recent races
    st.header("User controls")
    feel = st.selectbox("How do you feel?", ["good","ok","meh"], index=1)
    heat = st.selectbox("Heat", ["cool","moderate","hot"], index=0)
    recency_mode = st.select_slider(
        "Recency weighting",
        options=["off", "mild", "medium"],
        value="mild",
        help="Give recent races a bit more weight when building your pace curve."
    )
    # ensure that we update predictions / course map if user inputs change
    fingerprint = (aid_km_text, aid_units, heat, feel)
    if st.session_state.get('eta_fingerprint') != fingerprint:
        st.session_state.pop('eta_results', None)
        st.session_state['eta_fingerprint'] = fingerprint

# Build pace curves from Strava
if build_btn and tokens:
    try:
        acts = list_activities(tokens["access_token"], per_page=200)
        pace_df, used_df, meta = build_pace_curves_from_races(tokens["access_token"], acts, GRADE_BINS, max_activities=MAX_ACTIVITIES, recency_mode=recency_mode)
        # to share variables between reruns, assign them to session states
        st.session_state['pace_df'] = pace_df
        st.session_state['used_races'] = used_df
        st.session_state['model_meta'] = meta
        pc = st.session_state['pace_df']

        # persist if you already save CSVs
        pace_df.to_csv("data/pace_curves.csv", index=False)
        used_df.to_csv("data/used_races.csv", index=False)
        with open("data/model_meta.json", "w") as f:
            json.dump(meta, f)

        st.success("Built pace curves from your races ✅")
    except Exception as e:
        st.error(f"Failed: {e}")

with tab_race:
    st.subheader("Course map (GPX + aid stations)")

    if st.session_state.get('gpx_bytes') is None:
        st.info("Upload a GPX to view the route map.")
    else:
        try:
            df_map = parse_gpx(st.session_state['gpx_bytes'])
            # resample at fixed distance + smooth grade 
            df_res = resample_with_grade(df_map, step_m=STEP_LENGTH, window_m=STEP_WINDOW)

            # whole-course stats using the same hysteresis/resampling as segments
            length_km, gain_m, loss_m, min_ele, max_ele = segment_stats(df_res)

            # show summary above the map
            c1, c2, c3 = st.columns(3)
            c1.metric("Course length", f"{length_km:.1f} km")
            c2.metric("Total gain", f"{gain_m:.0f} m")
            c3.metric("Total loss", f"{loss_m:.0f} m")
            st.caption(f"Elevation range: {min_ele:.0f}–{max_ele:.0f} m")

            # Parse the aid-station list
            try:
                aid_km_list = parse_cumulative_dist(aid_km_text, aid_units)  # if you added the units toggle
            except NameError:
                aid_km_list = [float(x.strip()) for x in aid_km_text.split(",") if x.strip()]

            # Build the base map and fit to route bounds
            m = folium.Map(tiles="OpenStreetMap")
            route = list(zip(df_map['lat'].astype(float), df_map['lon'].astype(float)))
            if len(route) >= 2:
                folium.PolyLine(route, weight=3, opacity=0.9).add_to(m)
                min_lat, max_lat = float(df_map['lat'].min()), float(df_map['lat'].max())
                min_lon, max_lon = float(df_map['lon'].min()), float(df_map['lon'].max())
                m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

            # Add clustered aid-station markers (handles repeats like AS1/AS4 at same place)
            clusters = aid_station_markers(df_map, aid_km_list, cluster_radius_m=CLUSTER_RADIUS)
            for c in clusters:
                label = "/".join(c['labels'])
                km_list = ", ".join(f"{k:.1f} km" for k in sorted(c['kms']))
                folium.Marker(
                    location=[c['lat'], c['lon']],
                    tooltip=label,
                    popup=folium.Popup(html=f"<b>{label}</b><br/>{km_list}", max_width=250)
                ).add_to(m)

            # Render in Streamlit (display the map)
            st_folium(m, width=None, height=500)

        except Exception as e:
            st.warning(f"Could not render map: {e}")

    # Segments overview
    st.divider()
    st.subheader("Segments overview")
    if gpx_file is not None:
        try:
            df_raw = parse_gpx(st.session_state['gpx_bytes'])
            df = resample_with_grade(df_raw, step_m=STEP_LENGTH, window_m=STEP_WINDOW) # smoothing over gpx points
            aid_km_list = parse_cumulative_dist(aid_km_text, aid_units)  # returns km
            legs_idx = legs_from_aid_stations(df, aid_km_list)
            # names: Start -> AS1, AS1 -> AS2, ..., last -> Finish
            names_list = [f"AS{i + 1}" for i in range(len(legs_idx) - 1)] + ["Finish"]

            seg_rows = []
            for i, (a, b) in enumerate(legs_idx):
                seg = df.iloc[a:b + 1]
                length_km, gain_m, loss_m, min_ele, max_ele = segment_stats(seg)
                start_name = "Start" if i == 0 else names_list[i - 1] if i - 1 < len(names_list) else f"AS{i}"
                end_name = names_list[i] if i < len(names_list) else f"AS{i + 1}"
                title = f"{start_name} → {end_name}"
                with st.expander(f"{title}  •  {length_km:.1f} km  •  +{int(gain_m)}m / -{int(loss_m)}m"):
                    st.write(pd.DataFrame({
                        "Metric": ["Length (km)", "Elevation gain (m)", "Elevation loss (m)", "Min elev (m)",
                                   "Max elev (m)"],
                        "Value": [round(length_km, 1), int(gain_m), int(loss_m), int(min_ele), int(max_ele)]
                    }))
                    x_km = (seg['dist_m'] - seg['dist_m'].iloc[0]) / 1000.0
                    y_m = seg['ele_m']
                    fig, ax = plt.subplots(figsize=(5, 2))
                    ax.plot(x_km, y_m)
                    ax.set_xlabel("Distance (km)")
                    ax.set_ylabel("Elevation (m)")
                    ax.set_title(title)
                    st.pyplot(fig)
                    plt.close(fig)  # explicitly close the figure after Streamlit has copied it
                seg_rows.append({
                    "Segment": title,
                    "Km": round(length_km, 1),
                    "Gain_m": int(gain_m),
                    "Loss_m": int(loss_m),
                    "Min_ele_m": int(min_ele),
                    "Max_ele_m": int(max_ele)
                })
            if seg_rows:
                seg_df = pd.DataFrame(seg_rows)
                st.dataframe(seg_df, use_container_width=True)
                st.download_button("Download segments CSV", seg_df.to_csv(index=False).encode(),
                                   "segments_overview.csv", "text/csv")
        except Exception as e:
            st.warning(f"Could not render segments overview: {e}")

    st.divider()
    st.subheader("Predict ETAs")
    predict_btn = st.button("Run prediction", disabled=(gpx_file is None or pc is None))

    # Compute once on click, then cache in session_state
    if predict_btn and gpx_file is not None and pc is not None:
        try:
            # 0) sanity on pace curves vs bins
            if len(pc) != len(GRADE_BINS) - 1:
                st.error(f"Pace curves bins ({len(pc)}) do not match expected ({len(GRADE_BINS) - 1}). "
                         f"Try rebuilding curves from Strava.")
                raise RuntimeError("pace_df/bin mismatch")

            df = parse_gpx(st.session_state['gpx_bytes'])
            aid_km = parse_cumulative_dist(aid_km_text, aid_units)  # returns km

            # 1) build legs by GPX indices
            legs_idx = legs_from_aid_stations(df, aid_km)

            # 2) per-leg meters by fixed grade bins
            legs_meters = []
            for (a, b) in legs_idx:
                seg = df.iloc[a:b + 1]
                meters = distance_by_grade_bins(seg, GRADE_BINS)
                if meters.sum() > 1.0:
                    legs_meters.append(meters)

            n = len(legs_meters)
            if n == 0:
                st.error("Could not derive segments from GPX + aid-station list.")
                raise RuntimeError("no legs")

            # 3) derive each leg’s end km from the GPX itself (not the user list)
            leg_end_km = [float(df.iloc[b]['dist_m']) / 1000.0 for (_, b) in legs_idx][:n]

            # 4)  Course altitude factor (median elevation from GPX)
            H_course = float(np.nanmedian(df['ele_m'].to_numpy(dtype=float)))
            A_course = altitude_impairment_multiplicative(H_course)  # <= 1.0

            # 5) Use SEA-LEVEL curve and apply course altitude penalty
            speeds_sl = pc["speed_mps"].values
            sigmas = pc["sigma_rel"].values
            speeds = speeds_sl * A_course
            total_km = float(df["dist_m"].iloc[-1]) / 1000.0
            leg_ends_x = [min(1.0, k / max(total_km, EPSILON)) for k in leg_end_km]

            # 6) Use deterministic seed so identical inputs => identical ETAs
            seed_payload = str([GRADE_BINS,
                                list(np.round(speeds, 5)),
                                list(np.round(sigmas, 5)),
                                heat, feel,
                                list(np.round(leg_end_km, 3)),
                                round(total_km, 3)])
            seed = int(hashlib.sha256(seed_payload.encode()).hexdigest(), 16) % (2 ** 32)
            np.random.seed(seed)

            # 7) run sim (assumes simulate_etas returns P10/P50/P90)
            p10, p50, p90 = simulate_etas(legs_meters, speeds, sigmas, leg_ends_x, heat, feel, sims=MC_SIMS)

            # 8) Stamina scaling via personal Riegel exponent k ---
            meta = st.session_state.get('model_meta') or {}
            k = float(meta.get('riegel_k', DEFAULT_RIEGEL_K))
            Dref = meta.get('ref_distance_km', None)
            Tref = meta.get('ref_time_s', None)

            total_km = float(df["dist_m"].iloc[-1]) / 1000.0

            # Keep track of Riegel scaling actually applied (we only slow down; never speed up)
            riegel_applied = False
            riegel_scale = 1.0
            if Dref and Tref and total_km > 0:
                T_base50 = float(p50[-1])
                T_riegel = float(Tref) * (total_km / float(Dref)) ** float(k)
                s = float(T_riegel / max(EPSILON, T_base50))
                # only use Riegel factor to slow down longer efforts, not speed up short races
                if s > 1.0:
                    p10, p50, p90 = p10 * s, p50 * s, p90 * s
                    riegel_applied = True
                    riegel_scale = s

            # 8b) Add adjustment for very long races and make sure percentiles skew right
            p10, p50, p90, ultra_meta = apply_ultra_adjustments_progressive(p10, p50, p90, leg_ends_x)

            # 8c) Build a meta record for this prediction and cache it so it persists across reruns
            pred_meta = {
                "course_median_alt_m": float(H_course),
                "alt_speed_factor": float(A_course),  # multiplier on speeds (<1 means slower than sea level)
                "recency_mode": (st.session_state.get("model_meta") or {}).get("recency_mode", "mild"),
                "riegel_k": float(k),
                "riegel_applied": bool(riegel_applied),
                "riegel_scale_factor": float(riegel_scale),
                "ref_distance_km": float(Dref) if Dref else None,
                "ref_time_s": float(Tref) if Tref else None,
                "finish_time_p50_s": float(p50[-1]),
                **ultra_meta,
            }
            st.session_state["prediction_meta"] = pred_meta

            # 9) Annotate the UI with course altitude and the applied factor
            st.caption(
                f"Course median altitude ≈ {H_course:.0f} m → altitude factor {A_course:.2f}. Riegel k = {k:.2f}.")

            # 10) defend against any residual length drift
            L = min(n, len(leg_end_km), len(p10), len(p50), len(p90))
            leg_end_km = leg_end_km[:L]
            p10, p50, p90 = p10[:L], p50[:L], p90[:L]

            # 11) Auto-label: AS1..AS(L-1), Finish
            names = [f"AS{i + 1}" for i in range(L - 1)] + ["Finish"]

            out = pd.DataFrame({
                "Aid Station": names,
                "Km": [round(x, 1) for x in leg_end_km],
                "Arrival time (best guess)": [fmt(x) for x in p50],
                "10th percentile": [fmt(x) for x in p10],
                "90th percentile": [fmt(x) for x in p90],
            })

            st.session_state['eta_results'] = out

        except Exception as e:
            st.error(f"Error: {e}")

    # Always show last computed ETAs (if any), and download from the cached table
    res = st.session_state.get('eta_results')
    if res is not None:
        st.dataframe(res, use_container_width=True)
        st.download_button("Download ETAs CSV", res.to_csv(index=False).encode(), "eta_predictions.csv", "text/csv")

        # Button to clear cached ETAs
        if st.button("Clear ETAs cache"):
            st.session_state.pop('eta_results', None)
    else:
        st.info("No ETAs yet — click Run prediction.")


    meta = st.session_state.get("prediction_meta")
    if meta:
        # Friendly text for reference race
        ref_txt = "—"
        if meta.get("ref_distance_km") and meta.get("ref_time_s"):
            ref_txt = f"{meta['ref_distance_km']:.1f} km, {fmt(meta['ref_time_s'])}"

        # Speed factor -> slowdown in %
        a = float(meta.get("alt_speed_factor", 1.0))
        alt_pct = (1.0 - a) * 100.0  # % slower than sea level (speed domain)

        with st.expander("Model details for this prediction", expanded=False):
            st.markdown(
                f"""
    - **Course median altitude:** ~{meta['course_median_alt_m']:.0f} m  
      • **Altitude speed factor:** {a:.2f} (≈ {alt_pct:.0f}% slower vs sea level)
    - **Recency weighting:** `{meta.get('recency_mode','mild')}`
    - **Riegel exponent (k):** {meta['riegel_k']:.2f}  
      • **Reference race length and time:** {ref_txt}  
      • **Riegel scaling applied?** {'Yes' if meta['riegel_applied'] else 'No'}{f' (×{meta["riegel_scale_factor"]:.2f})' if meta['riegel_applied'] else ''}
    - **Ultra adjustments:** start after {meta['start_threshold_h']:.0f} h  
      • **Finish slowdown factor:** ×{meta['slow_factor_finish']:.2f}  
      • **Rest added by finish:** {fmt(meta['rest_added_finish_s'])}
    - **Predicted finish (P50):** {fmt(meta['finish_time_p50_s'])}
                """.strip()
            )

with tab_data:
    # Show used races (for sense-checking)
    st.subheader("Races used for prediction")
    used = st.session_state.get('used_races')
    if used is not None and len(used) > 0:
        df_used = used.copy()
        if 'id' in df_used.columns:
            df_used = df_used.drop_duplicates(subset='id')
            cols = ['name','date','distance_km']  # hide id in the table
        else:
            # Backward compatibility if your saved CSV didn't have 'id'
            df_used = df_used.drop_duplicates()
            cols = [c for c in df_used.columns if c in ('name','date','distance_km')]
            if not cols: cols = df_used.columns.tolist()
        st.dataframe(df_used[cols].sort_values('date', ascending=False).reset_index(drop=True),
                     use_container_width=True)
    else:
        st.info("No race list yet — connect Strava and click 'Build from my Strava races'.")


    # Show resulting pace curves (read-only)
    st.subheader("Derived pace curves (by grade bin)")
    if pc is not None:
        show = pc.copy().rename(columns={"speed_mps": "Average speed (in m/s)"})
        show.insert(0, "Grade bin", [f"{GRADE_BINS[i]}..{GRADE_BINS[i+1]}%" for i in range(len(GRADE_BINS)-1)])
        st.dataframe(show[["Grade bin","Average speed (in m/s)","sigma_rel"]], use_container_width=True)
    else:
        st.info("No pace curves yet.")

    meta = st.session_state.get('model_meta') or {}
    with st.expander("Model details (read-only)"):
        st.markdown(f"""
    - **Recency mode:** `{meta.get('recency_mode', 'mild')}`
    - **Altitude penalty α:** `0.06` per 1000 m above 300 m (hardcoded)
    - **Riegel exponent (k):** `{meta.get('riegel_k')}`
    - **Reference race length :** `{meta.get('ref_distance_km', '—')} km`
    - **Races used:** `{meta.get('n_races', 0)}`
    """)


