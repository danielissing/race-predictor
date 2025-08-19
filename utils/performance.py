"""
Performance modeling and adjustment functions.
Handles altitude effects, ultra-distance adjustments, and fatigue modeling.
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


def fatigue_multiplier(progress: float) -> float:
    """
    Calculate cumulative fatigue factor based on race progress.

    Models the progressive slowdown that occurs during long races due to
    glycogen depletion, muscle damage, and central fatigue.

    Args:
        progress: Fraction of race completed (0.0 to 1.0)

    Returns:
        Time multiplier (>= 1.0, where 1.0 = no fatigue)

    Fatigue progression:
        0-30% of race: No fatigue (multiplier = 1.00)
        30-60% of race: Mild fatigue (increases to 1.03)
        60-100% of race: Moderate fatigue (increases to 1.08)
    """
    if progress < 0.30:
        return 1.00
    elif progress < 0.60:
        # Linear increase from 1.00 to 1.03
        return 1.00 + (progress - 0.30) * (0.03 / 0.30)
    else:
        # Linear increase from 1.03 to 1.08
        return 1.03 + (progress - 0.60) * (0.05 / 0.40)


def apply_ultra_adjustments_progressive(p10, p50, p90, leg_ends_x):
    """
    Apply progressive adjustments for ultra-distance races.

    Ultra races (>6 hours) require special modeling for:
    1. Progressive slowdown due to cumulative fatigue
    2. Time spent at aid stations for refueling/rest
    3. Increased variability in later stages

    Args:
        p10, p50, p90: Percentile predictions (arrays)
        leg_ends_x: Fractional progress at each checkpoint (0.0 to 1.0)

    Returns:
        Tuple of (adjusted_p10, adjusted_p50, adjusted_p90, metadata_dict)

    The adjustments are progressive:
    - Start after START_TH_H hours (typically 6h)
    - Scale with race duration (longer = more adjustment)
    - Add both multiplicative slowdown and additive rest time
    """
    T50 = float(p50[-1])
    hours = T50 / config.SECONDS_PER_HOUR

    p10 = p10.copy()
    p50 = p50.copy()
    p90 = p90.copy()

    # No adjustments for shorter races
    if hours <= config.START_TH_H:
        meta = dict(
            start_threshold_h=config.START_TH_H,
            ultra_gamma=config.ULTRA_GAMMA,
            rest_per_24h_h=config.REST_PER_24H,
            slow_factor_finish=1.0,
            rest_added_finish_s=0.0,
        )
        return p10, p50, p90, meta

    slow_finish = 1.0
    rest_finish_s = 0.0

    # Apply progressive adjustments at each checkpoint
    for i, x in enumerate(leg_ends_x):
        # Estimated hours to this checkpoint
        hours_here = hours * float(x)

        # Calculate adjustment based on time beyond threshold
        # "blocks" = how many 24-hour periods beyond the threshold
        blocks = max(0.0, (hours_here - config.START_TH_H) / 24.0)

        # Multiplicative slowdown (fatigue accumulation)
        slow_factor = 1.0 + config.ULTRA_GAMMA * blocks

        # Additive rest time (aid station stops, brief rests)
        rest_seconds = (config.REST_PER_24H * config.SECONDS_PER_HOUR) * blocks

        # Apply adjustments with different weights for percentiles
        # P90 gets more rest time (pessimistic scenario)
        p10[i] = p10[i] * slow_factor + 0.25 * rest_seconds
        p50[i] = p50[i] * slow_factor + 0.50 * rest_seconds
        p90[i] = p90[i] * slow_factor + 1.00 * rest_seconds

        slow_finish = slow_factor
        rest_finish_s = rest_seconds

    # Additional right-tail skew for very long races (>18 hours)
    if hours >= config.SLEEP_CUTOFF:
        # Up to 10% additional slowdown for P90 in very long races
        skew = 1.0 + 0.1 * max(0.0, (hours - config.SLEEP_CUTOFF) / config.SLEEP_CUTOFF)
        p90 = p90 * skew

    meta = dict(
        start_threshold_h=config.START_TH_H,
        ultra_gamma=config.ULTRA_GAMMA,
        rest_per_24h_h=config.REST_PER_24H,
        slow_factor_finish=float(slow_finish),
        rest_added_finish_s=float(rest_finish_s),
    )

    return p10, p50, p90, meta


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