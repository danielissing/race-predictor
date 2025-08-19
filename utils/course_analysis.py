"""
Course-specific analysis functions.
Handles segmentation by aid stations and grade distribution analysis.
"""

import numpy as np
import pandas as pd

def legs_from_aid_stations(df: pd.DataFrame, aid_km: list) -> list:
    """
    Divide a course into segments based on aid station locations.

    Args:
        df: DataFrame with 'dist_m' column
        aid_km: List of cumulative distances to aid stations (in km)

    Returns:
        List of (start_idx, end_idx) tuples for each segment

    Example:
        For a 50km race with aid at 10km, 25km, 40km:
        Returns segments: Start->AS1, AS1->AS2, AS2->AS3, AS3->Finish
    """
    aid_m = [km * 1000.0 for km in aid_km]
    segments = []
    start_idx = 0

    for target_distance_m in aid_m:
        # Find the GPS point closest to this aid station distance
        end_idx = int(np.searchsorted(df['dist_m'].values, target_distance_m, side='right'))
        end_idx = min(max(end_idx, start_idx + 1), len(df) - 1)

        segments.append((start_idx, end_idx))
        start_idx = end_idx

    # Add final segment to finish if needed
    if segments and segments[-1][1] < len(df) - 1:
        segments.append((segments[-1][1], len(df) - 1))
    elif not segments:
        # No aid stations - entire course is one segment
        segments.append((0, len(df) - 1))

    return segments


def distance_by_grade_bins(segment: pd.DataFrame, grade_bins: list[float]) -> np.ndarray:
    """
    Analyze how much distance is covered at different grade ranges.

    This is crucial for pace prediction because running speed varies
    significantly with grade (uphill vs flat vs downhill).

    Args:
        segment: DataFrame with 'dist_m' and 'grade_pct' columns
        grade_bins: List of grade thresholds
                   e.g., [-10, -5, 0, 5, 10] creates 4 bins:
                   steep down [-10,-5], gentle down [-5,0],
                   gentle up [0,5], steep up [5,10]

    Returns:
        Array where each element is meters covered in that grade bin

    Example:
        For a 5km segment with bins [-10, -5, 0, 5, 10]:
        Returns [500, 1200, 2000, 1000, 300] meaning:
        - 500m of steep downhill
        - 1200m of gentle downhill
        - 2000m of flat
        - 1000m of gentle uphill
        - 300m of steep uphill
    """
    if segment.empty or len(segment) < 2:
        return np.zeros(len(grade_bins) - 1)

    # Calculate distance increments between consecutive points
    distance_increments = segment['dist_m'].diff().fillna(0.0).values
    grades = segment['grade_pct'].values

    # Assign each grade to a bin
    # np.digitize returns bin indices (1-indexed), so subtract 1
    bin_indices = np.digitize(grades, grade_bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(grade_bins) - 2)

    # Sum distance for each grade bin
    distance_by_bin = np.zeros(len(grade_bins) - 1)

    # Skip index 0 since diff() gives NaN for first point
    for i in range(1, len(segment)):
        bin_index = bin_indices[i]
        distance_by_bin[bin_index] += distance_increments[i]

    return distance_by_bin