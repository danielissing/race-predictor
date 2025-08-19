"""
UI display helper functions for the Race Time Predictor app.
All Streamlit-specific rendering logic lives here.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from utils.elevation import segment_stats
from utils.geo import aid_station_markers
from utils.persistence import fmt
import config


def display_course_details(course):
    """Renders the course map, metrics, and segment overview."""
    st.subheader("Course Map & Stats")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Course length", f"{course.total_km:.1f} km")
        st.metric("Total gain", f"{course.gain_m:.0f} m")
        st.metric("Total loss", f"{course.loss_m:.0f} m")
        st.caption(f"Elevation range: {course.min_ele:.0f}–{course.max_ele:.0f} m")

    with col2:
        m = folium.Map(tiles="OpenStreetMap")
        route = list(zip(course.df_raw['lat'], course.df_raw['lon']))
        folium.PolyLine(route, weight=4, opacity=0.8, color='blue').add_to(m)
        m.fit_bounds([[course.df_raw['lat'].min(), course.df_raw['lon'].min()],
                      [course.df_raw['lat'].max(), course.df_raw['lon'].max()]])
        clusters = aid_station_markers(course.df_raw, course.aid_km)
        for c in clusters:
            label = "/".join(c['labels'])
            km_list = ", ".join(f"{k:.1f} km" for k in sorted(c['kms']))
            folium.Marker(location=[c['lat'], c['lon']], tooltip=label,
                          popup=folium.Popup(html=f"<b>{label}</b><br/>{km_list}", max_width=250)).add_to(m)
        st_folium(m, width=None, height=400)


def display_segments_overview(course):
    """Displays segment breakdown with elevation profiles and download option."""
    try:
        names_list = [f"AS{i + 1}" for i in range(len(course.legs_idx) - 1)] + ["Finish"]
        seg_rows = []

        for i, (a, b) in enumerate(course.legs_idx):
            seg = course.df_res.iloc[a:b + 1]
            length_km, gain_m, loss_m, min_ele, max_ele = segment_stats(seg)
            start_name = "Start" if i == 0 else names_list[i - 1]
            end_name = names_list[i]
            title = f"{start_name} → {end_name}"

            with st.expander(f"{title}  •  {length_km:.1f} km  •  +{int(gain_m)}m / -{int(loss_m)}m"):
                fig, ax = plt.subplots(figsize=(7, 2.5))
                x_km = seg['dist_m'] / 1000.0
                y_m = seg['ele_m']
                ax.plot(x_km, y_m)
                ax.set_xlabel("Distance (km)")
                ax.set_ylabel("Elevation (m)")
                ax.set_title(title)
                st.pyplot(fig)
                plt.close(fig)

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
            st.download_button(
                "📥 Download segments CSV",
                seg_df.to_csv(index=False).encode(),
                "segments_overview.csv",
                "text/csv"
            )
    except Exception as e:
        st.warning(f"Could not render segments overview: {e}")


def display_prediction_results():
    """Display prediction results with download/clear buttons and model details."""
    if st.session_state.eta_results is None:
        st.info("No ETAs yet — click Run prediction.")
        return

    # Display the results table
    st.dataframe(st.session_state.eta_results, use_container_width=True)

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download ETAs CSV",
            st.session_state.eta_results.to_csv(index=False).encode(),
            "eta_predictions.csv",
            "text/csv"
        )
    with col2:
        if st.button("Clear ETAs cache"):
            st.session_state.eta_results = None
            st.session_state.prediction_meta = None
            st.rerun()

    # Display model details if available
    display_prediction_metadata()


def display_prediction_metadata():
    """Display detailed model metadata in an expander."""
    meta = st.session_state.get("prediction_meta")
    if not meta:
        return

    # Format reference race text
    ref_txt = "—"
    if meta.get("ref_distance_km") and meta.get("ref_time_s"):
        ref_txt = f"{meta['ref_distance_km']:.1f} km, {fmt(meta['ref_time_s'])}"

    # Calculate altitude slowdown percentage
    a = float(meta.get("alt_speed_factor", 1.0))
    alt_pct = (1.0 - a) * 100.0

    with st.expander("Model details for this prediction", expanded=False):
        st.markdown(f"""
- **Course median altitude:** ~{meta['course_median_alt_m']:.0f} m  
  • **Altitude speed factor:** {a:.2f} (≈ {alt_pct:.0f}% slower vs sea level)
- **Recency weighting:** `{meta.get('recency_mode', 'mild')}`
- **Riegel exponent (k):** {meta['riegel_k']:.2f}  
  • **Reference race length and time:** {ref_txt}  
  • **Riegel scaling applied?** {'Yes' if meta['riegel_applied'] else 'No'}{f' (×{meta["riegel_scale_factor"]:.2f})' if meta['riegel_applied'] else ''}
- **Ultra adjustments:** start after {meta.get('start_threshold_h', 0):.0f} h  
  • **Finish slowdown factor:** ×{meta.get('slow_factor_finish', 1):.2f}  
  • **Rest added by finish:** {fmt(meta.get('rest_added_finish_s', 0))}
- **Predicted finish (P50):** {fmt(meta['finish_time_p50_s'])}
        """.strip())


def display_pace_model_races(pace_model):
    """Display the races used to build the pace model."""
    st.subheader("Races Used for Prediction")

    if pace_model.used_races is None or len(pace_model.used_races) == 0:
        st.info("No races found in the model.")
        return

    display_df = pace_model.used_races.copy()
    if 'id' in display_df.columns:
        display_df = display_df.drop_duplicates(subset='id')

    cols = ['name', 'date', 'distance_km']
    cols = [c for c in cols if c in display_df.columns]

    if cols:
        st.dataframe(
            display_df[cols].sort_values('date', ascending=False).reset_index(drop=True),
            use_container_width=True
        )


def display_pace_curves(pace_model):
    """Display the derived pace curves by grade bin."""
    st.subheader("Derived Pace Curves (by Grade Bin)")

    if pace_model.pace_df is None:
        st.info("No pace curves available.")
        return

    show = pace_model.pace_df.copy()
    show.insert(0, "Grade bin",
                [f"{config.GRADE_BINS[i]}..{config.GRADE_BINS[i + 1]}%"
                 for i in range(len(config.GRADE_BINS) - 1)])
    show = show.rename(columns={"speed_mps": "Average speed (m/s)"})

    st.dataframe(
        show[["Grade bin", "Average speed (m/s)", "sigma_rel"]],
        use_container_width=True,
        column_config={
            "Average speed (m/s)": st.column_config.NumberColumn(
                "Average speed (m/s)", format="%.2g"
            ),
            "sigma_rel": st.column_config.NumberColumn(
                "sigma_rel", format="%.2g"
            ),
        },
    )


def display_model_metadata(pace_model):
    """Display model metadata in an expander."""
    with st.expander("Model details (read-only)"):
        st.markdown(f"""
- **Recency mode:** `{pace_model.meta.get('recency_mode', 'mild')}`
- **Altitude penalty α:** {config.ELEVATION_IMPAIRMENT} per 1000 m above 300 m 
- **Riegel exponent (k):** `{pace_model.riegel_k:.2f}`
- **Reference race length:** `{pace_model.ref_distance_km or '—'} km`
- **Races used:** `{pace_model.meta.get('n_races', 0)}`
        """)