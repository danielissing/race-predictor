"""
Elevation analysis and resampling utilities.
Handles elevation gain/loss calculations and grade smoothing.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import config

def segment_stats(
        segment_df: pd.DataFrame,
        resample_step_m: float = config.STEP_LENGTH,
        min_step_m: float = float(config.DEFAULT_SMOOTH_WINDOW)
) -> Tuple[float, float, float, float, float]:
    """
    Calculate comprehensive statistics for a course segment.

    Uses intelligent resampling and hysteresis filtering to get accurate
    elevation gain/loss numbers that match what runners actually experience.

    Args:
        segment_df: DataFrame with 'dist_m' and 'ele_m' columns
        resample_step_m: Distance interval for resampling (default 20m)
        min_step_m: Minimum elevation change to count as real gain/loss (default 3m)

    Returns:
        Tuple of (length_km, gain_m, loss_m, min_elevation_m, max_elevation_m)

    Note:
        The hysteresis filtering prevents counting GPS noise as elevation change.
        For example, if GPS shows +1m, -1m, +1m, -1m (noise), this counts as 0 gain/loss.
    """
    if not _is_valid_segment_for_stats(segment_df):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    distances = segment_df['dist_m'].to_numpy(dtype=float)
    elevations = segment_df['ele_m'].to_numpy(dtype=float)

    if len(distances) < 2 or np.all(np.isnan(elevations)):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Resample elevations at consistent distance intervals
    resampled_elevations = _resample_elevations_for_stats(distances, elevations, resample_step_m)

    if np.all(np.isnan(resampled_elevations)):
        length_km = (distances[-1] - distances[0]) / 1000.0
        return length_km, 0.0, 0.0, 0.0, 0.0

    # Calculate elevation gain/loss with hysteresis filtering
    gain_m, loss_m = _calculate_elevation_changes_with_hysteresis(resampled_elevations, min_step_m)

    # Calculate summary statistics
    length_km = (distances[-1] - distances[0]) / 1000.0
    min_elevation = float(np.nanmin(resampled_elevations))
    max_elevation = float(np.nanmax(resampled_elevations))

    return length_km, gain_m, loss_m, min_elevation, max_elevation


def resample_with_grade(
        df: pd.DataFrame,
        step_m: float = config.STEP_LENGTH,
        window_m: float = config.STEP_WINDOW
) -> pd.DataFrame:
    """
    Resample GPS track to regular intervals and calculate smoothed grades.

    This is essential for accurate pace modeling because:
    1. GPS devices record points at irregular intervals
    2. Raw elevation data is noisy and needs smoothing
    3. Grade calculations need consistent spacing

    Args:
        df: DataFrame with 'dist_m' and 'ele_m' columns
        step_m: Distance between resampled points (e.g., every 20m)
        window_m: Smoothing window size (e.g., 100m for grade calculation)

    Returns:
        DataFrame with evenly spaced points and 'grade_pct' column

    Example:
        Input:  Irregular GPS points at 0m, 7m, 18m, 31m...
        Output: Regular points at 0m, 20m, 40m, 60m... with smoothed grades
    """
    resampled_df = _resample_to_regular_distances(df, step_m)
    return _add_smoothed_grades(resampled_df, window_m, step_m)


def _is_valid_segment_for_stats(segment_df: pd.DataFrame) -> bool:
    """Check if segment has required columns and data."""
    return (
            not segment_df.empty and
            'dist_m' in segment_df.columns and
            'ele_m' in segment_df.columns
    )


def _resample_elevations_for_stats(
        distances: np.ndarray,
        elevations: np.ndarray,
        step_m: float
) -> np.ndarray:
    """
    Resample elevation data at consistent distance intervals.

    This creates a regular grid of elevation samples, which is needed
    for accurate gain/loss calculations.
    """
    start_distance, end_distance = distances[0], distances[-1]

    # If segment is too short for resampling, use endpoints only
    if end_distance - start_distance < step_m:
        sample_distances = np.array([start_distance, end_distance])
    else:
        sample_distances = np.arange(start_distance, end_distance, step_m)

    # Linear interpolation between GPS points
    return np.interp(sample_distances, distances, elevations)


def _calculate_elevation_changes_with_hysteresis(
        elevations: np.ndarray,
        min_threshold_m: float
) -> Tuple[float, float]:
    """
    Calculate elevation gain and loss with intelligent hysteresis filtering.

    Hysteresis Algorithm:
    - Track upward and downward movement in accumulators
    - Only commit to gain/loss when accumulated change exceeds threshold
    - This filters out GPS noise while preserving real elevation changes

    Example:
        If threshold is 3m:
        - Climbing 2m then descending 1m = no gain recorded (noise)
        - Climbing 5m then descending 1m = 5m gain recorded (real climb)

    Args:
        elevations: Array of elevation values at regular intervals
        min_threshold_m: Minimum change to count as real (not noise)

    Returns:
        Tuple of (total_gain_m, total_loss_m)
    """
    total_gain = 0.0
    total_loss = 0.0
    upward_accumulator = 0.0
    downward_accumulator = 0.0

    previous_elevation = elevations[0]

    for current_elevation in elevations[1:]:
        elevation_change = current_elevation - previous_elevation

        if elevation_change >= 0:  # Moving upward
            upward_accumulator += elevation_change

            # If we were going down and accumulated enough loss, commit it
            if downward_accumulator <= -min_threshold_m:
                total_loss += -downward_accumulator
            downward_accumulator = 0.0

        else:  # Moving downward
            downward_accumulator += elevation_change

            # If we were going up and accumulated enough gain, commit it
            if upward_accumulator >= min_threshold_m:
                total_gain += upward_accumulator
            upward_accumulator = 0.0

        previous_elevation = current_elevation

    # Commit any remaining accumulated changes that meet threshold
    if upward_accumulator >= min_threshold_m:
        total_gain += upward_accumulator
    if downward_accumulator <= -min_threshold_m:
        total_loss += -downward_accumulator

    return total_gain, total_loss


def _resample_to_regular_distances(df: pd.DataFrame, step_m: float) -> pd.DataFrame:
    """
    Convert irregularly spaced GPS points to evenly spaced intervals.

    GPS devices record points based on time or when turning, not distance.
    This creates a consistent grid for analysis.
    """
    if len(df) < 2:
        return df.copy()

    distances = df["dist_m"].to_numpy(dtype=float)
    elevations = df["ele_m"].to_numpy(dtype=float)

    # Create regular distance grid
    start_distance = distances[0]
    end_distance = distances[-1]
    regular_distances = np.arange(start_distance, end_distance + step_m, step_m)

    # Interpolate elevations at regular points
    interpolated_elevations = np.interp(regular_distances, distances, elevations)

    return pd.DataFrame({
        "dist_m": regular_distances,
        "ele_m": interpolated_elevations
    })


def _add_smoothed_grades(
        df: pd.DataFrame,
        window_m: float,
        step_m: float
) -> pd.DataFrame:
    """
    Calculate smoothed grade percentages from elevation data.

    Two-stage smoothing process:
    1. Median filter removes outliers (GPS errors)
    2. Mean filter provides final smoothing

    This gives grades that match what runners actually experience,
    filtering out GPS noise while preserving real terrain features.
    """
    elevations = df["ele_m"].to_numpy(dtype=float)

    # Calculate smoothing window size in number of points
    smoothing_points = max(3, int(round(window_m / step_m)))
    if smoothing_points % 2 == 0:  # Ensure odd number for centering
        smoothing_points += 1

    # Two-stage smoothing
    elevation_series = pd.Series(elevations)
    smoothed_elevations = (
        elevation_series
        .rolling(smoothing_points, center=True, min_periods=1)
        .median()  # Remove outliers
        .rolling(smoothing_points, center=True, min_periods=1)
        .mean()  # Final smoothing
        .to_numpy()
    )

    # Calculate grade using numerical gradient
    # Grade = rise/run * 100%
    grade_percent = np.gradient(smoothed_elevations, step_m) * 100.0

    result_df = df.copy()
    result_df["grade_pct"] = grade_percent
    return result_df