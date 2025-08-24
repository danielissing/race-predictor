"""
Performance modeling and adjustment functions.
"""

import numpy as np
from datetime import datetime, timezone
import config

def altitude_impairment_multiplicative(
        mean_elevation: float,
        alpha: float = config.ELEVATION_IMPAIRMENT,
        elevation_threshold: float = config.ELEVATION_IMPAIRMENT_THRESHOLD
) -> float:
    """
    Calculate speed reduction factor due to altitude.

    At high elevations, reduced oxygen availability decreases running performance.
    This function models that effect based on scientific literature.

    Args:
        mean_elevation: Average elevation of the course/segment (meters)
        alpha: Impairment coefficient (default 0.06 = 6% per 1000m above threshold)
        elevation_threshold: Elevation where impairment begins (default 300m)

    Returns:
        Speed multiplier between 0 and 1 (1.0 = no impairment)

    Example:
        At sea level (0m): returns 1.0 (no impairment)
        At 2000m: returns ~0.90 (10% slower)
        At 3000m: returns ~0.84 (16% slower)
    """
    if mean_elevation <= elevation_threshold:
        return 1.0

    elevation_above_threshold_km = (mean_elevation - elevation_threshold) / 1000.0
    return 1.0 - alpha * elevation_above_threshold_km


def recency_weight(start_date_str: str, mode: str = "mild") -> float:
    """
    Calculate weight for a race based on how recent it was.

    More recent races are given higher weight when building pace models,
    as they better reflect current fitness.

    Args:
        start_date_str: ISO format date string (e.g., "2024-01-15T10:00:00Z")
        mode: Weighting mode
            - "off": No recency weighting (always returns 1.0)
            - "mild": Gentle decay (half-life defined in config)
            - "medium": Faster decay (half the mild half-life)

    Returns:
        Weight between 0 and 1 (1.0 = full weight for recent races)

    Example:
        For "mild" mode with 12-month half-life:
        - Race from today: weight = 1.0
        - Race from 6 months ago: weight ≈ 0.71
        - Race from 12 months ago: weight = 0.5
        - Race from 24 months ago: weight = 0.25
    """
    if mode == "off":
        return 1.0

    now = datetime.now(timezone.utc)
    race_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00")).astimezone(timezone.utc)

    # Calculate age in months
    months_old = max(0.0, (now - race_date).days / config.MONTH_LENGTH)

    # Determine half-life based on mode
    half_life = config.MILD_HALF_LIFE if mode == "mild" else config.MILD_HALF_LIFE / 2

    # Exponential decay formula
    decay_rate = np.log(2) / half_life  # per month
    return float(np.exp(-decay_rate * months_old))


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """
    Calculate weighted percentile of data.

    Standard percentiles treat all data points equally. Weighted percentiles
    give different importance to different points (e.g., based on recency).

    Args:
        x: Data values
        w: Weights for each value (non-negative)
        q: Percentile to compute (0 to 100)

    Returns:
        Weighted percentile value

    Example:
        x = [10, 20, 30]
        w = [1, 2, 1]  # Middle value has double weight
        weighted_percentile(x, w, 50) -> closer to 20 than regular median
    """
    if len(x) == 0:
        return float("nan")

    # Sort data by value
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]

    # Calculate cumulative distribution
    cumsum = np.cumsum(w_sorted)
    if cumsum[-1] == 0:
        # All weights are zero - fall back to regular median
        return float(np.median(x))

    # Normalize to [0, 1]
    cumulative_prob = cumsum / cumsum[-1]

    # Interpolate to find percentile
    return float(np.interp(q / 100.0, cumulative_prob, x_sorted))