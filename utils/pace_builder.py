"""
Build personalized pace curves from historical race data.
Integrates with Strava to analyze past performances.
"""
# packages
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
#local imports
from utils.strava import get_activity_streams, is_run, is_race
from utils.performance import altitude_impairment_multiplicative, recency_weight, weighted_percentile
import config


def build_pace_curves_from_races(
        access_token: str,
        activities: List[Dict[str, Any]],
        bins: list,
        max_activities: int = config.MAX_ACTIVITIES,
        recency_mode: str = "mild",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Build personalized pace curves from Strava race history.

    This is the main function that analyzes all your past races to create
    a pace model that predicts how fast you can run at different grades.

    Key features:
    1. Altitude normalization - converts all paces to sea-level equivalent
    2. Recency weighting - recent races count more than old ones
    3. Personal Riegel exponent - captures your individual endurance profile
    4. Grade-specific analysis - different speeds for uphill/flat/downhill

    Args:
        access_token: Strava API access token
        activities: List of Strava activities
        bins: Grade bins (e.g., [-10, -5, 0, 5, 10, 15] for 5 bins)
        max_activities: Maximum number of races to analyze
        recency_mode: How to weight recent races ("off", "mild", "medium")

    Returns:
        Tuple of:
        - curves_df: DataFrame with columns [lower_pct, upper_pct, speed_mps, sigma_rel]
        - used_races_df: DataFrame of races used in the model
        - meta: Dictionary with model metadata (Riegel k, reference race, etc.)
    """
    n_bins = len(bins) - 1

    # Collect samples for each grade bin
    speed_samples_by_bin = [[] for _ in range(n_bins)]
    weight_samples_by_bin = [[] for _ in range(n_bins)]

    # Filter to only race activities
    races = _filter_and_deduplicate_races(activities)

    used_race_metadata = []

    # Process each race
    for activity in races[:max_activities]:
        race_data = _process_single_race(
            access_token, activity, bins, n_bins, recency_mode,
            speed_samples_by_bin, weight_samples_by_bin
        )

        if race_data:
            used_race_metadata.append(race_data)

    # Create DataFrames
    used_races_df = _create_used_races_dataframe(used_race_metadata)
    curves_df = _create_pace_curves_dataframe(
        bins, speed_samples_by_bin, weight_samples_by_bin
    )

    # Fit personal Riegel exponent
    riegel_k, ref_distance_km, ref_time_s = _fit_riegel_exponent(used_races_df)

    # Build metadata dictionary
    meta = {
        "alpha": config.ELEVATION_IMPAIRMENT,
        "recency_mode": recency_mode,
        "riegel_k": float(riegel_k),
        "ref_distance_km": float(ref_distance_km) if ref_distance_km else None,
        "ref_time_s": float(ref_time_s) if ref_time_s else None,
        "n_races": int(len(used_races_df)) if used_races_df is not None else 0,
    }

    return curves_df, used_races_df, meta


def _filter_and_deduplicate_races(activities: List[Dict]) -> List[Dict]:
    """Filter to only races and remove duplicates."""
    races = [a for a in activities if is_run(a) and is_race(a)]

    # Deduplicate by activity ID
    seen = set()
    unique_races = []
    for race in races:
        race_id = race.get("id")
        if race_id not in seen:
            seen.add(race_id)
            unique_races.append(race)

    return unique_races


def _process_single_race(
        access_token: str,
        activity: Dict,
        bins: list,
        n_bins: int,
        recency_mode: str,
        speed_samples_by_bin: list,
        weight_samples_by_bin: list
) -> Dict[str, Any]:
    """
    Process a single race activity and extract grade-specific pace data.

    Returns race metadata if successfully processed, None otherwise.
    """
    activity_id = activity.get("id")
    streams = get_activity_streams(access_token, activity_id)

    if not streams:
        return None

    # Extract stream data
    stream_data = _extract_stream_data(streams)
    if not stream_data:
        return None

    dist_m, vel, grd_arr, alt, mov = stream_data

    # Calculate altitude adjustment
    median_alt = float(np.nanmedian(alt)) if alt is not None else 0.0
    altitude_factor = altitude_impairment_multiplicative(median_alt)

    # Calculate recency weight
    race_weight = recency_weight(activity.get("start_date", ""), recency_mode)

    # Process each GPS point
    used_any = _process_gps_points(
        dist_m, vel, grd_arr, mov,
        altitude_factor, race_weight,
        bins, n_bins,
        speed_samples_by_bin, weight_samples_by_bin
    )

    if not used_any:
        return None

    # Return race metadata
    return {
        "id": activity_id,
        "name": activity.get("name", "(unnamed)"),
        "date": (activity.get("start_date", "") or "")[:10],
        "distance_km": round(activity.get("distance", 0) / 1000.0, 2),
        "elapsed_time_s": int(activity.get("elapsed_time") or 0),
        "median_alt_m": median_alt,
        "weight": round(race_weight, 3),
    }


def _extract_stream_data(streams: Dict) -> tuple:
    """Extract and process GPS stream data from Strava."""
    dist = streams.get("distance", {}).get("data")
    if dist is None or len(dist) < 2:
        return None

    dist_m = np.array(dist, dtype=float)

    # Get velocity (prefer smooth velocity, calculate if needed)
    vs = streams.get("velocity_smooth", {}).get("data")
    if vs is not None:
        vel = np.array(vs, dtype=float)
    else:
        time_s = streams.get("time", {}).get("data")
        if time_s is None:
            return None
        vel = _calculate_velocity_from_time(dist_m, time_s)

    # Get grade (calculate from altitude if needed)
    grd = streams.get("grade_smooth", {}).get("data")
    alt = streams.get("altitude", {}).get("data")

    if grd is not None:
        grd_arr = np.array(grd, dtype=float)
    elif alt is not None:
        grd_arr = _calculate_grade_from_altitude(dist_m, alt)
    else:
        grd_arr = np.zeros_like(dist_m)

    # Get moving status
    moving = streams.get("moving", {}).get("data")
    if moving is None:
        moving = (vel > 0.2).astype(int).tolist()
    mov = np.array(moving, dtype=float)

    return dist_m, vel, grd_arr, alt, mov


def _calculate_velocity_from_time(dist_m: np.ndarray, time_s: list) -> np.ndarray:
    """Calculate velocity from distance and time arrays."""
    t = np.array(time_s, dtype=float)
    dt = np.diff(t, prepend=t[0])
    dd = np.diff(dist_m, prepend=dist_m[0])
    dt = np.clip(dt, 1e-3, None)
    return dd / dt


def _calculate_grade_from_altitude(dist_m: np.ndarray, alt: list) -> np.ndarray:
    """Calculate grade percentage from altitude data."""
    alt_m = np.array(alt, dtype=float)
    dd = np.diff(dist_m, prepend=dist_m[0])
    da = np.diff(alt_m, prepend=alt_m[0])
    dd = np.clip(dd, 1e-3, None)
    return (da / dd) * 100.0


def _process_gps_points(
        dist_m, vel, grd_arr, mov,
        altitude_factor, race_weight,
        bins, n_bins,
        speed_samples_by_bin, weight_samples_by_bin
) -> bool:
    """
    Process GPS points and bin speeds by grade.

    Returns True if any valid data was processed.
    """
    # Calculate distance increments
    dd = np.diff(dist_m, prepend=dist_m[0])

    # Filter valid points (moving, positive distance)
    mask = (vel > 0.2) & (mov > 0.5) & (dd > 0)
    dd = dd * mask

    # Bin grades
    bin_idx = np.digitize(grd_arr, bins, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    used_any = False

    for i in range(1, len(dist_m)):
        if dd[i] <= 0:
            continue

        bin_index = int(bin_idx[i])

        # Convert to sea-level equivalent speed
        sea_level_speed = float(
            np.clip(vel[i] / max(altitude_factor, config.EPSILON),
                    config.SEA_LEVEL_CLIP_LOW,
                    config.SEA_LEVEL_CLIP_HIGH)
        )

        # Add to appropriate bin with weight
        speed_samples_by_bin[bin_index].append(sea_level_speed)
        weight_samples_by_bin[bin_index].append(float(dd[i] * race_weight))
        used_any = True

    return used_any


def _create_used_races_dataframe(race_metadata: List[Dict]) -> pd.DataFrame:
    """Create DataFrame of races used in the model."""
    df = pd.DataFrame(race_metadata)
    if not df.empty:
        df = (df
              .drop_duplicates(subset="id")
              .sort_values("date", ascending=False)
              .reset_index(drop=True))
    return df


def _create_pace_curves_dataframe(
        bins: list,
        speed_samples_by_bin: list,
        weight_samples_by_bin: list
) -> pd.DataFrame:
    """
    Create pace curves DataFrame from binned samples.

    For each grade bin, calculates:
    - Median speed (50th percentile)
    - Variability (from 10th and 90th percentiles)
    """
    rows = []

    for i in range(len(bins) - 1):
        speeds = np.array(speed_samples_by_bin[i], dtype=float)
        weights = np.array(weight_samples_by_bin[i], dtype=float)

        if len(speeds) == 0 or np.sum(weights) == 0:
            # No data for this bin - use default values
            rows.append({
                "lower_pct": bins[i],
                "upper_pct": bins[i + 1],
                "speed_mps": 1.2,  # Default slow jog speed
                "sigma_rel": config.SIGMA_REL_DEFAULT
            })
        else:
            # Calculate weighted statistics
            median_speed = weighted_percentile(speeds, weights, 50)
            p10_speed = weighted_percentile(speeds, weights, 10)
            p90_speed = weighted_percentile(speeds, weights, 90)

            # Relative variability (coefficient of variation)
            relative_sigma = (p90_speed - p10_speed) / max(config.EPSILON, 2 * median_speed)

            rows.append({
                "lower_pct": bins[i],
                "upper_pct": bins[i + 1],
                "speed_mps": float(median_speed),
                "sigma_rel": float(np.clip(relative_sigma,
                                           config.SIGMA_REL_LOW,
                                           config.SIGMA_REL_HIGH))
            })

    return pd.DataFrame(rows)


def _fit_riegel_exponent(used_races_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Fit personal Riegel exponent from race history.

    The Riegel formula: T = T_ref * (D / D_ref) ^ k

    Where k captures how well you maintain pace over longer distances:

    Returns:
        Tuple of (k, reference_distance_km, reference_time_s)
    """
    if used_races_df.empty:
        return config.DEFAULT_RIEGEL_K, None, None

    # Extract valid races
    distances = used_races_df["distance_km"].to_numpy(dtype=float)
    times = used_races_df["elapsed_time_s"].to_numpy(dtype=float)
    weights = used_races_df["weight"].to_numpy(dtype=float)

    valid_mask = (distances > 0) & (times > 0)
    if np.sum(valid_mask) < 2:
        return config.DEFAULT_RIEGEL_K, None, None

    # Weighted linear regression in log-log space
    # log(T) = log(a) + k * log(D)
    X = np.log(distances[valid_mask])
    Y = np.log(times[valid_mask])
    W = weights[valid_mask]

    # Calculate weighted regression coefficients
    k = _weighted_linear_regression_slope(X, Y, W)

    # Choose reference race near median distance
    ref_distance_km, ref_time_s = _choose_reference_race(
        distances[valid_mask],
        times[valid_mask],
        W
    )

    return k, ref_distance_km, ref_time_s


def _weighted_linear_regression_slope(X, Y, W) -> float:
    """Calculate slope of weighted linear regression."""
    WX = W * X
    WY = W * Y
    S = np.sum(W)
    SX = np.sum(WX)
    SY = np.sum(WY)
    SXX = np.sum(W * X * X)
    SXY = np.sum(W * X * Y)

    denominator = (S * SXX - SX * SX)
    if denominator > 0:
        return float((S * SXY - SX * SY) / denominator)
    return config.DEFAULT_RIEGEL_K


def _choose_reference_race(distances, times, weights) -> Tuple[float, float]:
    """Choose a reference race near the weighted median distance."""
    order = np.argsort(distances)
    d_sorted = distances[order]
    t_sorted = times[order]
    w_sorted = weights[order]

    # Find weighted median
    cumulative_weights = np.cumsum(w_sorted)
    cumulative_weights = cumulative_weights / cumulative_weights[-1]

    median_idx = int(np.searchsorted(cumulative_weights, 0.5))
    median_idx = min(median_idx, len(d_sorted) - 1)

    return float(d_sorted[median_idx]), float(t_sorted[median_idx])