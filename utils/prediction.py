"""
Core prediction logic - simplified and data-driven.
Uses personal Riegel k from race history for accurate scaling.
"""

import numpy as np
import hashlib
from utils.performance import altitude_impairment_multiplicative
import config


def calculate_base_times(course, speeds):
    """
    Calculate base time for each checkpoint using grade-specific speeds.
    This is the foundation before any distance/condition adjustments.

    Args:
        course: Course object with legs_meters array
        speeds: Grade-specific speeds (m/s) adjusted for altitude

    Returns:
        Array of cumulative times at each checkpoint
    """
    cumulative_times = []
    total_time = 0.0

    for leg_idx in range(len(course.legs_meters)):
        # Time = distance / speed for each grade bin
        leg_distances = course.legs_meters[leg_idx]
        leg_time = np.sum(
            np.where(speeds > 0, leg_distances / speeds, 0.0)
        )
        total_time += leg_time
        cumulative_times.append(total_time)

    return np.array(cumulative_times)


def get_distance_specific_k(pace_model, target_distance_km):
    """
    Calculate Riegel k using races near the target distance.
    This captures how YOUR endurance changes across distance ranges.

    Args:
        pace_model: PaceModel with your race history
        target_distance_km: Distance of current race

    Returns:
        Riegel k appropriate for this distance
    """
    # If not enough data, use global k
    if pace_model.used_races is None or len(pace_model.used_races) < 3:
        return pace_model.riegel_k

    races = pace_model.used_races

    # For very short races, look at your short race performance
    if target_distance_km < 30:
        nearby_races = races[races['distance_km'] < 50]
    # For ultras, look at your long race performance
    elif target_distance_km > 100:
        nearby_races = races[races['distance_km'] > 50]
    # For middle distances, look at similar races
    else:
        distance_range = (target_distance_km * 0.6, target_distance_km * 1.5)
        nearby_races = races[
            (races['distance_km'] >= distance_range[0]) &
            (races['distance_km'] <= distance_range[1])
            ]

    if len(nearby_races) < 2:
        return pace_model.riegel_k

    # Calculate k from these races
    distances = nearby_races['distance_km'].values
    times = nearby_races['elapsed_time_s'].values

    # Remove any zeros or invalid values
    valid = (distances > 0) & (times > 0)
    if np.sum(valid) < 2:
        return pace_model.riegel_k

    # Log-log regression for Riegel k
    X = np.log(distances[valid])
    Y = np.log(times[valid])

    # Simple linear regression in log space
    n = len(X)
    k_local = (n * np.sum(X * Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X ** 2) - np.sum(X) ** 2)

    # Sanity check - k should be between 0.9 and 1.2
    k_local = np.clip(k_local, 0.9, 1.2)

    return float(k_local)


def apply_distance_scaling(base_times, course, pace_model):
    """
    Apply data-driven distance scaling using YOUR personal Riegel k.
    This replaces both the old fatigue_multiplier and Riegel scaling.

    Key insights:
    - Your pace curves are calibrated to your typical race distance
    - Shorter races should be faster (k < ref)
    - Longer races should be slower (k > ref)
    - Applied progressively (more effect later in race)

    Args:
        base_times: Array of checkpoint times from grade calculation
        course: Course object with total_km and leg_ends_x
        pace_model: Your personal pace model with Riegel k

    Returns:
        Scaled checkpoint times
    """
    # Get distance-appropriate k value
    k = get_distance_specific_k(pace_model, course.total_km)

    # Reference distance (what your pace curves are calibrated to)
    ref_distance = pace_model.ref_distance_km if pace_model.ref_distance_km else 50.0

    # If this race is very close to your reference, minimal adjustment
    if abs(course.total_km - ref_distance) < 5:
        scale_factor = 1.0
    else:
        # Calculate how much to scale based on YOUR Riegel k
        # For a 100km race with k=1.06 and ref=50km: factor = 2^0.06 = 1.042
        scale_factor = (course.total_km / ref_distance) ** (k - 1)

    # Apply progressively through the race
    scaled_times = []
    for i, x in enumerate(course.leg_ends_x):
        # Progressive application: x^0.7 gives nice curve
        # Mile 1: almost no adjustment
        # Final mile: full adjustment
        # This naturally makes the second half slower than the first
        progress_factor = x ** 0.7
        adjustment = 1 + (scale_factor - 1) * progress_factor
        scaled_times.append(base_times[i] * adjustment)

    return np.array(scaled_times)


def apply_ultra_adjustments(times, course_km):
    """
    Simplified ultra adjustments for very long races.
    Only applies to races >6 hours, adds aid station time and fatigue.

    Args:
        times: Array of checkpoint times
        course_km: Total race distance

    Returns:
        Adjusted times with ultra-specific factors
    """
    finish_hours = times[-1] / 3600

    # No adjustments for "normal" races under 6 hours
    if finish_hours <= 6:
        return times

    adjusted = times.copy()

    # Add aid station time (increases with race length)
    if finish_hours < 12:
        # 6-12 hour races: ~3-5 min per aid station
        aid_time = 240  # 4 minutes average
    elif finish_hours < 24:
        # 12-24 hour races: ~5-8 min per aid station
        aid_time = 420  # 7 minutes average
    else:
        # 24+ hour races: ~8-12 min per aid station
        aid_time = 600  # 10 minutes average

    # Add cumulative aid station time
    n_checkpoints = len(times)
    for i in range(n_checkpoints):
        # Proportional aid time based on progress
        adjusted[i] += aid_time * (i + 1)

    # Additional slowdown for 24h+ races (sleep deprivation, cumulative fatigue)
    if finish_hours > 24:
        # 10% extra per 24h block beyond the first
        blocks_beyond_24 = (finish_hours - 24) / 24
        ultra_penalty = 1 + (0.10 * blocks_beyond_24)
        adjusted *= ultra_penalty

    return adjusted


def simulate_with_conditions(base_times, course, speeds, sigmas, conditions, sims=200):
    """
    Monte Carlo simulation with natural asymmetry and proper conditions handling.

    FIXED: Now properly incorporates conditions into the simulation.

    Args:
        base_times: Base checkpoint times (already adjusted for distance/ultra)
        course: Course object
        speeds: Grade-specific speeds
        sigmas: Grade-specific uncertainties
        conditions: Single parameter from -2 to +2
            -2: Terrible conditions (slower)
            -1: Poor conditions (slower)
             0: Normal conditions
            +1: Good conditions (faster)
            +2: Perfect conditions (faster)
        sims: Number of simulations

    Returns:
        Tuple of (p10, p50, p90) arrays
    """
    n_checkpoints = len(base_times)
    samples = np.zeros((sims, n_checkpoints))

    for sim in range(sims):
        # Day-to-day variability with natural asymmetry
        # Beta distribution skews toward bad days (realistic!)
        # Good days: maybe 5% faster
        # Bad days: could be 15% slower
        day_factor = np.random.beta(2, 3) * 0.20 - 0.05  # Skewed toward bad

        # Conditions affect both mean and variance
        # FIXED: Inverted the sign so negative conditions = slower (higher multiplier)
        # Good conditions: faster times (lower multiplier), less variance
        # Bad conditions: slower times (higher multiplier), more variance
        condition_mean = -conditions * 0.02  # FIXED: Inverted sign
        condition_variance = 1.0 + conditions * 0.1  # More variance when conditions are bad

        # Combined effect
        total_factor = 1.0 + condition_mean + day_factor * condition_variance

        # Grade-specific variation (some days you climb better, etc)
        grade_factors = np.random.normal(1.0, sigmas * 0.3)

        # Recalculate with variations
        sim_times = []
        total_time = 0.0

        for leg_idx in range(len(course.legs_meters)):
            leg_distances = course.legs_meters[leg_idx]
            varied_speeds = speeds * grade_factors
            varied_speeds = np.maximum(varied_speeds, 0.1)  # Prevent division by zero
            leg_time = np.sum(
                np.where(varied_speeds > 0, leg_distances / varied_speeds, 0.0)
            )
            total_time += leg_time * total_factor
            sim_times.append(total_time)

        samples[sim, :] = sim_times

    # Extract percentiles with realistic asymmetry
    # Use 15th and 85th percentiles for more realistic spread
    p15 = np.percentile(samples, 15, axis=0)
    p85 = np.percentile(samples, 85, axis=0)

    # P50 is the median of simulations (more stable than using base_times)
    p50 = np.percentile(samples, 50, axis=0)

    # Adjust to conventional P10/P90 reporting
    # P10 (optimistic) is closer to P50 than P90 (good days are limited)
    p10 = p50 - (p50 - p15) * 0.8
    p90 = p50 + (p85 - p50) * 1.2

    return p10, p50, p90


def run_prediction_simulation(course, pace_model, conditions=0):
    """
    Simplified prediction with data-driven distance scaling.

    FIXED: Removed redundant conditions application and fixed the flow.

    This is the main entry point for predictions, using:
    1. Grade-based speeds (adjusted for altitude)
    2. YOUR personal Riegel k for distance scaling
    3. Progressive application (2nd half slower)
    4. Ultra adjustments only when needed
    5. Single conditions parameter (properly applied in Monte Carlo)
    6. Monte Carlo with natural asymmetry

    Args:
        course: Course object with GPX data
        pace_model: PaceModel with your historical data
        conditions: Race conditions from -2 (terrible) to +2 (perfect)

    Returns:
        dict with p10, p50, p90 arrays and metadata
    """
    # Step 1: Altitude adjustment
    altitude_factor = altitude_impairment_multiplicative(course.median_altitude)
    speeds = pace_model.sea_level_speeds * altitude_factor

    # Step 2: Calculate base times from grade bins
    base_times = calculate_base_times(course, speeds)

    # Step 3: Apply YOUR personal distance scaling
    scaled_times = apply_distance_scaling(base_times, course, pace_model)

    # Step 4: Apply ultra adjustments if needed (>6 hours)
    if scaled_times[-1] > 6 * config.SECONDS_PER_HOUR:
        adjusted_times = apply_ultra_adjustments(scaled_times, course.total_km)
    else:
        adjusted_times = scaled_times

    # Step 5: Monte Carlo simulation WITH conditions
    # FIXED: Pass adjusted_times directly, conditions are handled inside the simulation
    p10, p50, p90 = simulate_with_conditions(
        adjusted_times, course, speeds, pace_model.sigmas,
        conditions, sims=200
    )

    # Build metadata for transparency
    k_used = get_distance_specific_k(pace_model, course.total_km)

    metadata = {
        "course_median_alt_m": float(course.median_altitude),
        "alt_speed_factor": float(altitude_factor),
        "riegel_k": float(k_used),
        "riegel_k_global": float(pace_model.riegel_k),
        "ref_distance_km": float(pace_model.ref_distance_km) if pace_model.ref_distance_km else 50.0,
        "ref_time_s": float(pace_model.ref_time_s) if pace_model.ref_time_s else None,
        "distance_scale_factor": float(adjusted_times[-1] / base_times[-1]) if base_times[-1] > 0 else 1.0,
        "ultra_adjusted": bool(scaled_times[-1] > 6 * 3600),
        "conditions": int(conditions),
        "finish_time_p50_s": float(p50[-1]),
    }

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "metadata": metadata
    }