"""
GPX file parsing and basic track processing.
Handles reading GPX files and converting them to clean DataFrames.
"""

import io
import numpy as np
import pandas as pd
import gpxpy
from typing import List, Tuple
from utils.geo import haversine_m
import config


def parse_gpx(file_bytes: bytes, smooth_window: int = config.DEFAULT_SMOOTH_WINDOW) -> pd.DataFrame:
    """
    Parse GPX file and return DataFrame with coordinates, distances, and smoothed grades.

    Args:
        file_bytes: Raw GPX file bytes
        smooth_window: Window size for grade smoothing (use 1 for no smoothing)

    Returns:
        DataFrame with columns: lat, lon, ele_m, dist_m, grade_pct

    Raises:
        ValueError: If GPX file cannot be parsed or contains no valid tracks
    """
    gpx_data = _parse_gpx_data(file_bytes)
    points = _extract_gps_points(gpx_data)
    df = _create_base_dataframe(points)
    df = _interpolate_elevation(df)
    df = _calculate_grades(df, smooth_window)
    return df


def parse_cumulative_dist(text: str, units: str) -> list[float]:
    """
    Parse comma-separated distance values and convert to kilometers.

    Args:
        text: Comma-separated distance values (e.g., "10, 21.1, 32, 42.2")
        units: Either "km" or "mi" 

    Returns:
        List of distances in kilometers

    Example:
        parse_cumulative_dist("10, 21.1", "km") -> [10.0, 21.1]
        parse_cumulative_dist("6.2, 13.1", "mi") -> [10.0, 21.1]
    """
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if units == "mi":
        return [v * config.MILES_TO_KM for v in vals]
    return vals


def _parse_gpx_data(file_bytes: bytes):
    """Parse GPX bytes into gpxpy object with error handling."""
    try:
        gpx_content = file_bytes.decode('utf-8', errors='ignore')
        gpx_data = gpxpy.parse(io.StringIO(gpx_content))
    except Exception as e:
        raise ValueError(f"Failed to parse GPX file: {e}")

    if not gpx_data.tracks:
        raise ValueError("No tracks found in GPX file")

    return gpx_data


def _extract_gps_points(gpx_data) -> List[Tuple[float, float, float, float]]:
    """
    Extract GPS coordinates and calculate cumulative distances from GPX data.

    Returns:
        List of tuples: (latitude, longitude, elevation_m, cumulative_distance_m)
    """
    points = []
    cumulative_distance = 0.0
    last_point = None

    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                # Calculate distance from previous point
                if last_point:
                    distance = haversine_m(
                        last_point.latitude, last_point.longitude,
                        point.latitude, point.longitude
                    )
                    cumulative_distance += distance if not np.isnan(distance) else 0.0

                # Store point data
                elevation = point.elevation if point.elevation is not None else np.nan
                points.append((point.latitude, point.longitude, elevation, cumulative_distance))
                last_point = point

    if not points:
        raise ValueError("No valid GPS points found in GPX file")

    return points


def _create_base_dataframe(points: List[Tuple[float, float, float, float]]) -> pd.DataFrame:
    """Create DataFrame from GPS points."""
    return pd.DataFrame(points, columns=['lat', 'lon', 'ele_m', 'dist_m'])


def _interpolate_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing elevation data using interpolation."""
    df = df.copy()
    df['ele_m'] = df['ele_m'].interpolate().bfill().ffill()
    return df


def _calculate_grades(df: pd.DataFrame, smooth_window: int) -> pd.DataFrame:
    """
    Calculate and smooth grade percentages.

    Grade is the slope expressed as a percentage:
    - 10% grade = 10m elevation gain per 100m horizontal distance
    - Positive grades = uphill, negative = downhill
    """
    df = df.copy()

    # Calculate distance and elevation changes between consecutive points
    distance_deltas = df['dist_m'].diff().fillna(0.0).clip(lower=config.EPSILON)
    elevation_deltas = df['ele_m'].diff().fillna(0.0)

    # Calculate raw grade percentages
    grade_percent = (elevation_deltas / distance_deltas) * 100.0

    # Apply smoothing if requested (reduces GPS noise)
    if smooth_window > 1:
        grade_percent = grade_percent.rolling(
            window=smooth_window,
            center=True,
            min_periods=1
        ).mean()

    df['grade_pct'] = grade_percent
    return df