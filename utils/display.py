"""
UI display helper functions for the Race Time Predictor app.
All Streamlit-specific rendering logic lives here.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from utils.elevation import segment_stats
from utils.geo import aid_station_markers
from utils.performance import altitude_impairment_multiplicative
import config
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

def format_seconds(sec: int) -> str:
    """Formats seconds into H:MM format."""
    sec = int(sec)
    h = sec // config.SECONDS_PER_HOUR
    m = (sec % config.SECONDS_PER_HOUR) // config.MINUTES_PER_HOUR
    return f"{h:d}:{m:02d}"

def mps_to_mpk(mps: float) -> str:
    """Convert speed in m/s (e.g. 3.14) to minutes per kilometer (e.g. '5:19')."""
    if mps <= 0:
        raise ValueError("Speed must be greater than zero")

    # total time in seconds to cover 1 km
    total_seconds = 1000 / mps

    minutes = int(total_seconds // 60)
    seconds = int(round(total_seconds % 60))

    # handle case where rounding pushes seconds to 60
    if seconds == 60:
        minutes += 1
        seconds = 0

    return f"{minutes}:{seconds:02d}"

def sigma_mps_to_sigma_mpk(speed: float, sigma_s: float) -> str:
    """
    Convert error in speed (m/s) to error in pace (min/km),
    and return it as a 'mm:ss' string
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if sigma_s < 0:
        sigma_s = abs(sigma_s)

    # sigma in minutes per km
    sigma_mpk_min = (1000 / (60 * speed**2)) * sigma_s

    # split into minutes and seconds
    minutes = int(sigma_mpk_min)
    seconds = int(round((sigma_mpk_min - minutes) * 60))

    # handle rounding to 60 seconds
    if seconds == 60:
        minutes += 1
        seconds = 0

    return f"±{minutes}:{seconds:02d}"

def display_course_details(course):
    """Renders the course map, metrics, and segment overview."""
    st.subheader("Course Map & Stats")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Course length", f"{course.total_km:.1f} km")
        st.metric("Total gain", f"{course.gain_m:.0f} m")
        st.metric("Total loss", f"{course.loss_m:.0f} m")
        st.metric("Elevation range", f"{course.min_ele:.0f}–{course.max_ele:.0f} m")

    with col2:
        # Get valid bounds first
        lat_min = float(course.df_raw['lat'].min())
        lat_max = float(course.df_raw['lat'].max())
        lon_min = float(course.df_raw['lon'].min())
        lon_max = float(course.df_raw['lon'].max())

        # Check if bounds are valid
        if pd.isna(lat_min) or pd.isna(lat_max) or pd.isna(lon_min) or pd.isna(lon_max):
            st.error("Invalid GPS coordinates in GPX file")
            return

        # Calculate center and appropriate zoom level
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        # Estimate zoom level based on bounds
        lat_diff = lat_max - lat_min
        lon_diff = lon_max - lon_min
        max_diff = max(lat_diff, lon_diff)

        # Rough zoom calculation
        if max_diff > 1:
            zoom_start = 8
        elif max_diff > 0.5:
            zoom_start = 9
        elif max_diff > 0.1:
            zoom_start = 11
        elif max_diff > 0.05:
            zoom_start = 12
        else:
            zoom_start = 13

        # Initialize map with center and zoom
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="OpenStreetMap"
        )

        # Add the route
        route = list(zip(course.df_raw['lat'], course.df_raw['lon']))
        folium.PolyLine(route, weight=4, opacity=0.8, color='blue').add_to(m)

        # Fit bounds after adding the route
        m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

        # Add aid station markers
        clusters = aid_station_markers(course.df_raw, course.aid_km)
        for c in clusters:
            label = "/".join(c['labels'])
            km_list = ", ".join(f"{k:.1f} km" for k in sorted(c['kms']))
            folium.Marker(
                location=[c['lat'], c['lon']],
                tooltip=label,
                popup=folium.Popup(html=f"<b>{label}</b><br/>{km_list}", max_width=250)
            ).add_to(m)

        st_folium(m, width=None, height=400, returned_objects=[])


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
                # Create the enhanced elevation plot
                fig = create_elevation_profile(seg, title)
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


def create_elevation_profile(seg, title):
    """
    Create an enhanced elevation profile with grade-based coloring.
    Simpler version without legend for cleaner appearance.

    Args:
        seg: DataFrame segment with dist_m, ele_m, and grade_pct columns
        title: Title for the plot

    Returns:
        matplotlib figure
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    # Extract data
    x_km = seg['dist_m'].values / 1000.0
    y_m = seg['ele_m'].values

    # Calculate grades if not present
    if 'grade_pct' in seg.columns:
        grades = seg['grade_pct'].values
    else:
        # Calculate grade from elevation change
        distance_diff = np.diff(seg['dist_m'].values)
        elevation_diff = np.diff(seg['ele_m'].values)
        grades_raw = np.zeros(len(seg))
        grades_raw[1:] = np.where(distance_diff > 0,
                                  (elevation_diff / distance_diff) * 100,
                                  0)
        grades = grades_raw

    # Apply Gaussian smoothing for more organic transitions
    # Sigma scales with segment length for appropriate smoothing
    sigma = max(2, min(20, len(grades) // 30))
    if len(grades) > 10:
        grades_smooth = gaussian_filter1d(grades, sigma=sigma)
    else:
        grades_smooth = grades

    # Create figure with better aspect ratio
    fig, ax = plt.subplots(figsize=(8, 3))

    # Create a continuous color mapping based on grade
    # Using a more intuitive color scheme
    def grade_to_color(grade):
        """Convert grade percentage to color with smooth transitions."""
        abs_grade = abs(grade)

        if abs_grade < 2:
            # Flat - green
            return '#2ECC71'
        elif abs_grade < 5:
            # Gentle - light green to yellow blend
            blend = (abs_grade - 2) / 3
            return interpolate_color('#2ECC71', '#F39C12', blend)
        elif abs_grade < 10:
            # Moderate - yellow to orange blend
            blend = (abs_grade - 5) / 5
            return interpolate_color('#F39C12', '#E67E22', blend)
        elif abs_grade < 15:
            # Steep - orange to red blend
            blend = (abs_grade - 10) / 5
            return interpolate_color('#E67E22', '#E74C3C', blend)
        else:
            # Very steep - red to dark red
            blend = min(1, (abs_grade - 15) / 10)
            return interpolate_color('#E74C3C', '#C0392B', blend)

    def interpolate_color(color1, color2, blend):
        """Interpolate between two hex colors."""
        # Convert hex to RGB
        c1 = [int(color1[i:i + 2], 16) for i in (1, 3, 5)]
        c2 = [int(color2[i:i + 2], 16) for i in (1, 3, 5)]
        # Interpolate
        c = [int(c1[i] + (c2[i] - c1[i]) * blend) for i in range(3)]
        # Convert back to hex
        return '#{:02x}{:02x}{:02x}'.format(*c)

    # Fill with grade-based colors in segments
    # Use larger chunks for cleaner appearance
    chunk_size = max(1, len(x_km) // 100)  # Aim for ~100 color segments max

    for i in range(0, len(x_km) - 1, chunk_size):
        end_idx = min(i + chunk_size + 1, len(x_km))
        if end_idx > i + 1:
            # Get average grade for this chunk
            avg_grade = np.mean(grades_smooth[i:end_idx])
            color = grade_to_color(avg_grade)

            # Fill this section
            ax.fill_between(x_km[i:end_idx],
                            y_m[i:end_idx],
                            np.min(y_m) - 100,  # Extend well below
                            color=color,
                            alpha=0.4,
                            edgecolor='none')

    # Plot the elevation profile line on top
    ax.plot(x_km, y_m, color='#2C3E50', linewidth=2, zorder=3)

    # Styling
    ax.set_xlabel("Distance (km)", fontsize=10, color='#34495E')
    ax.set_ylabel("Elevation (m)", fontsize=10, color='#34495E')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#2C3E50')

    # Set y-axis limits with padding
    y_range = y_m.max() - y_m.min()
    y_padding = max(20, y_range * 0.15)  # At least 20m padding
    ax.set_ylim(y_m.min() - y_padding, y_m.max() + y_padding)

    # Subtle grid
    ax.grid(True, alpha=0.2, linestyle=':', color='#95A5A6')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    # Add subtle gradient indicator text
    ax.text(0.02, 0.98, 'Green=Flat • Yellow=Moderate • Red=Steep',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            color='#7F8C8D',
            alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()

    return fig


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

def display_pace_model_races(pace_model):
    """Display the races used to build the pace model."""
    st.subheader("Races used for prediction")

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


def display_model_metadata(pace_model):
    """Display model metadata in an expander."""
    with st.expander("Model details (read-only)"):
        st.markdown(f"""
- **Recency mode:** `{pace_model.meta.get('recency_mode', 'mild')}`
- **Altitude penalty α:** `{config.ELEVATION_IMPAIRMENT}` per 1000 m above 300 m 
- **Global Riegel exponent (using data from all races):** `{pace_model.riegel_k:.2f}`
- **Median race length:** `{pace_model.ref_distance_km or '—'} km`
- **Races used:** `{pace_model.meta.get('n_races', 0)}`
        """)


def plot_pace_curves(pace_model, current_altitude_m=None):
    """
    Create a visualization of pace curves showing speed vs grade.

    Args:
        pace_model: PaceModel object with pace_df containing speeds
        current_altitude_m: Optional altitude to show adjusted speeds for

    Returns:
        matplotlib figure
    """
    if pace_model is None or pace_model.pace_df is None:
        return None

    # Extract data from pace model
    pace_df = pace_model.pace_df

    # Calculate grade midpoints for each bin
    grade_midpoints = (pace_df['lower_pct'].values + pace_df['upper_pct'].values) / 2

    # Get sea-level speeds (these are already normalized to sea level in pace_df)
    sea_level_speeds = pace_df['speed_mps'].values

    # Convert to pace (min/km)
    sea_level_paces = 1000 / (sea_level_speeds * 60)

    # Calculate average training altitude from historical races for "raw" speeds
    avg_training_altitude = None
    if pace_model.used_races is not None and 'median_alt_m' in pace_model.used_races.columns:
        avg_training_altitude = pace_model.used_races['median_alt_m'].mean()

    # Create figure 
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot sea-level pace
    line1 = ax1.plot(grade_midpoints, sea_level_paces,
                     'b-o', linewidth=2, markersize=8,
                     label='Sea level')

    # Add error bars for uncertainty (using sigma_rel)
    if 'sigma_rel' in pace_df.columns:
        uncertainties = (1000/ (60* sea_level_speeds**2)) * pace_df['sigma_rel'].values
        ax1.fill_between(grade_midpoints,
                         sea_level_paces - uncertainties,
                         sea_level_paces + uncertainties,
                         alpha=0.2, color='blue')

    lines_for_legend = line1

    # Add "raw" pace curve at average training altitude
    if avg_training_altitude is not None and avg_training_altitude > 0:
        raw_altitude_factor = altitude_impairment_multiplicative(avg_training_altitude)
        raw_speeds = sea_level_speeds * raw_altitude_factor
        raw_paces = 1000 / (raw_speeds * 60)

        line_raw = ax1.plot(grade_midpoints, raw_paces,
                            'g-^', linewidth=1.5, markersize=5, alpha=0.7,
                            label=f'Avg altitude in your race history ({avg_training_altitude:.0f}m)')
        lines_for_legend += line_raw

    # If current altitude provided, show adjusted speeds for the race
    if current_altitude_m is not None and current_altitude_m > 0:
        altitude_factor = altitude_impairment_multiplicative(current_altitude_m)
        altitude_correction = altitude_factor -1 
        adjusted_speeds = sea_level_speeds * altitude_factor
        adjusted_paces = 1000 / (adjusted_speeds * 60)

        line2 = ax1.plot(grade_midpoints, adjusted_paces,
                         'r--o', linewidth=2, markersize=6,
                         label=f'Current race mean altitude ({current_altitude_m:.0f}m)')
        lines_for_legend += line2

        # Add annotation showing the adjustment
        ax1.annotate(f'Race altitude correction: {altitude_correction:.0%}',
                     xy=(0.02, 0.98), xycoords='axes fraction',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format primary y-axis (speed)
    ax1.set_xlabel('Grade (%)', fontsize=12)
    ax1.set_ylabel('Pace (min/km)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    # Add vertical line at 0% grade
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    # Add horizontal line at typical flat speed
    flat_speed_idx = np.argmin(np.abs(grade_midpoints))
    flat_speed = sea_level_paces[flat_speed_idx]
    ax1.axhline(y=flat_speed, color='gray', linestyle=':', alpha=0.5)

    # Title and legend
    plt.title('Personal Pace Curve: Speed vs Grade', fontsize=14, fontweight='bold')
    ax1.legend(handles=lines_for_legend, loc='upper right')

    # Add informative text
    fig.text(0.12, 0.02,
             'Note: Pace is normalized to sea level, uncertainties are at one sigma level.',
             fontsize=9, style='italic')

    plt.tight_layout()
    return fig


def plot_pace_comparison(pace_model, reference_speeds=None):
    """
    Compare your pace curve to a reference (e.g., world class runner).

    Args:
        pace_model: Your PaceModel object
        reference_speeds: Optional dict with reference speeds by grade

    Returns:
        matplotlib figure
    """
    if pace_model is None or pace_model.pace_df is None:
        return None

    pace_df = pace_model.pace_df
    grade_midpoints = (pace_df['lower_pct'].values + pace_df['upper_pct'].values) / 2
    your_speeds = pace_df['speed_mps'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot your speeds
    ax.plot(grade_midpoints, your_speeds, 'b-o', linewidth=2,
            markersize=8, label='Your Speed')

    # Add reference if provided
    if reference_speeds:
        ref_speeds = [reference_speeds.get(g, your_speeds[i])
                      for i, g in enumerate(grade_midpoints)]
        ax.plot(grade_midpoints, ref_speeds, 'g--s', linewidth=1.5,
                markersize=6, label='Reference Runner', alpha=0.7)

        # Show percentage differences
        pct_diff = (your_speeds - np.array(ref_speeds)) / np.array(ref_speeds) * 100
        for i, (g, pct) in enumerate(zip(grade_midpoints, pct_diff)):
            if abs(pct) > 5:  # Only show significant differences
                ax.annotate(f'{pct:+.0f}%',
                            xy=(g, your_speeds[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)

    ax.set_xlabel('Grade (%)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.title('Pace Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig


def display_pace_curve_analysis(pace_model, current_course=None):
    """
    Display comprehensive pace curve analysis in Streamlit.

    Args:
        pace_model: PaceModel object
        current_course: Optional Course object for altitude adjustment
    """
    if pace_model is None or pace_model.pace_df is None:
        st.info("No pace model available. Build one from the sidebar.")
        return

    st.subheader("🏃 Your Personal Pace Curve")

    # Determine altitude for adjustment
    altitude_m = None
    if current_course:
        altitude_m = current_course.median_altitude
        st.info(f"Showing speeds adjusted for current course altitude: {altitude_m:.0f}m")

    # Create and display the main pace curve plot
    fig = plot_pace_curves(pace_model, altitude_m)
    if fig:
        st.pyplot(fig)
        plt.close(fig)

    # Analysis section
    with st.expander("📊 Pace Curve Analysis", expanded=True):
        pace_df = pace_model.pace_df

        col1, col2, col3 = st.columns(3)

        # Calculate uphill/downhill efficiency
        steep_down_idx = 0
        steep_up_idx = len(pace_df) - 1

        with col1:
            # Find flat pace (using sea-level speed)
            flat_idx = len(pace_df) // 2  # Approximate
            flat_speed = pace_df.iloc[flat_idx]['speed_mps']
            flat_pace = mps_to_mpk(flat_speed)
            st.metric("Flat pace (sea level)", f"{flat_pace} min/km")

        with col2:
            down_speed = pace_df.iloc[steep_down_idx]['speed_mps']
            down_diff = (down_speed / flat_speed - 1) * 100
            st.metric("Downhill difference (vs flat)", f"{down_diff:.0f}%")

        with col3:
            up_speed = pace_df.iloc[steep_up_idx]['speed_mps']
            up_diff = (up_speed / flat_speed - 1) * 100
            st.metric("Uphill difference (vs flat)", f"{up_diff:.0f}%")

        # Show raw data table
        st.subheader("Personal pace data (Sea level normalized)")
        display_df = pace_df.copy()
        display_df['grade_range'] = display_df.apply(
            lambda r: f"{r['lower_pct']:.0f}% to {r['upper_pct']:.0f}%", axis=1
        )
        display_df['pace_min_km'] = display_df.apply(lambda r: mps_to_mpk(r['speed_mps']), axis=1)

        # Fix uncertainty calculation - sigma_rel is relative, need to multiply by speed first
        display_df['uncertainty'] = display_df.apply(
            lambda r: sigma_mps_to_sigma_mpk(r['speed_mps'], r['speed_mps'] * r['sigma_rel']), axis=1
        )

        st.dataframe(
            display_df[['grade_range', 'pace_min_km', 'uncertainty']],
            use_container_width=True,
            hide_index=True
        )