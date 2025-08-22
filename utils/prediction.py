"""
Core prediction logic - simplified and data-driven.
Uses personal Riegel k from race history for accurate scaling.
WITH DEBUG OUTPUT for troubleshooting ultra predictions
"""

import numpy as np
import hashlib
from utils.performance import altitude_impairment_multiplicative
import config

# Debug flag - set to False to disable all debug output
DEBUG = True


def debug_print(msg):
    """Helper function for debug output"""
    if DEBUG:
        print(f"[DEBUG] {msg}")


def format_time(seconds):
    """Format seconds to readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    else:
        return f"{minutes}m {secs:02d}s"


def _is_flat_race(course):
    # Detect if this is likely a flat road race
    gain_per_km = course.gain_m / course.total_km if course.total_km > 0 else 0
    return gain_per_km < config.ROAD_GAIN_PER_KM


def calculate_base_times(course, speeds):
    """
    Calculate base time for each checkpoint using grade-specific speeds.
    Special handling for flat road races to use primarily flat pace.

    Args:
        course: Course object with legs_meters array
        speeds: Grade-specific speeds (m/s) adjusted for altitude

    Returns:
        Array of cumulative times at each checkpoint
    """
    # Detect if this is likely a flat road race
    is_flat_road = _is_flat_race(course)

    debug_print("=" * 60)
    debug_print("STAGE 1: CALCULATE BASE TIMES")
    debug_print(f"Course: {course.total_km:.1f}km, {course.gain_m:.0f}m gain")
    debug_print(f"Is flat road race: {is_flat_road}")
    debug_print(f"Number of checkpoints: {len(course.legs_meters)}")

    cumulative_times = []
    total_time = 0.0

    for leg_idx in range(len(course.legs_meters)):
        leg_distances = course.legs_meters[leg_idx]

        if is_flat_road:
            # For flat road races, use mostly the flat pace
            # This avoids issues with grade calculation noise
            flat_bin_idx = len(speeds) // 2  # Middle bin is usually flat
            # Weight heavily toward flat pace
            weighted_speeds = speeds.copy()
            for i in range(len(weighted_speeds)):
                # Blend toward flat pace based on distance from center
                dist_from_flat = abs(i - flat_bin_idx)
                weight = max(0.3, 1.0 - dist_from_flat * 0.15)
                weighted_speeds[i] = speeds[flat_bin_idx] * weight + speeds[i] * (1 - weight)

            leg_time = np.sum(
                np.where(weighted_speeds > 0, leg_distances / weighted_speeds, 0.0)
            )
        else:
            # Normal calculation for trail races
            leg_time = np.sum(
                np.where(speeds > 0, leg_distances / speeds, 0.0)
            )

        total_time += leg_time
        cumulative_times.append(total_time)

        # Debug output for key checkpoints
        if leg_idx < len(course.leg_ends_x):
            dist_km = course.leg_ends_x[leg_idx] * course.total_km
            # Show first few checkpoints and any near 10k
            if leg_idx < 3 or (9 <= dist_km <= 11) or leg_idx == len(course.legs_meters) - 1:
                pace_min_km = (total_time / 60) / dist_km if dist_km > 0 else 0
                debug_print(
                    f"  Checkpoint {leg_idx}: {dist_km:.1f}km -> {format_time(total_time)} ({pace_min_km:.1f} min/km)")

    debug_print(f"Base finish time: {format_time(cumulative_times[-1])}")
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
    if target_distance_km < config.SHORT_DISTANCE_KM:
        nearby_races = races[races['distance_km'] < int(config.DEFAULT_DISTANCE_KM)]
    # For ultras, look at your long race performance
    elif target_distance_km > config.MEDIUM_DISTANCE_KM:
        nearby_races = races[races['distance_km'] > int(config.DEFAULT_DISTANCE_KM)]
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

    # Sanity check - k should not be crazy
    k_local = np.clip(k_local, config.MIN_RIEGEL_K, config.MAX_RIEGEL_K)

    return float(k_local)


def apply_distance_scaling(base_times, course, pace_model):
    """
    Apply data-driven distance scaling using YOUR personal Riegel k.

    Key insights:
    - Your pace curves are calibrated to your typical race distance
    - Shorter races should be faster (k < ref)
    - Longer races should be slower (k > ref)
    - Road races need less adjustment than trail races

    Args:
        base_times: Array of checkpoint times from grade calculation
        course: Course object with total_km and leg_ends_x
        pace_model: Your personal pace model with Riegel k

    Returns:
        Scaled checkpoint times
    """
    debug_print("=" * 60)
    debug_print("STAGE 2: APPLY DISTANCE SCALING")

    # Detect if this is likely a flat road race
    is_flat_road = _is_flat_race(course)

    # Get distance-appropriate k value
    k = get_distance_specific_k(pace_model, course.total_km)

    # For road races, use a more conservative Riegel_k (less variation with distance)
    if is_flat_road:
        # Road racing is more predictable, k closer to 1.0
        k = 1.0 + (k - 1.0) * config.ROAD_K_DAMPEN  #

    # Reference distance (what your pace curves are calibrated to)
    ref_distance = pace_model.ref_distance_km if pace_model.ref_distance_km else config.DEFAULT_DISTANCE_KM

    debug_print(f"Riegel k: {k:.3f}")
    debug_print(f"Reference distance: {ref_distance:.1f} km")
    debug_print(f"Target distance: {course.total_km:.1f} km")
    debug_print(f"Distance ratio: {course.total_km / ref_distance:.2f}x")

    # If this race is very close to your reference, minimal adjustment
    if abs(course.total_km - ref_distance) < 5:
        scale_factor = 1.0
        debug_print("Race close to reference distance - no scaling")
    else:
        distance_ratio = course.total_km / ref_distance

        # Different dampening based on race type and direction
        if course.total_km < ref_distance:
            # Shorter races
            if is_flat_road:
                # Road races: very conservative speedup
                effective_k = 1 + (k - 1) * config.ROAD_K_DAMPEN
            elif course.total_km < config.SHORT_DISTANCE_KM:  # Short trail races
                effective_k = 1 + (k - 1) * 0.5  # 50% effect
            else:  # Medium trail races shorter than ref
                effective_k = 1 + (k - 1) * 0.7  # 70% effect
            scale_factor = distance_ratio ** (effective_k - 1)
            debug_print(f"Shorter than ref - effective k: {effective_k:.3f}")
        else:
            # Longer races
            if is_flat_road:
                # Road ultras still predictable
                effective_k = 1 + (k - 1) * 0.7
            else:
                # Trail ultras use full k
                effective_k = k
            scale_factor = distance_ratio ** (effective_k - 1)
            debug_print(f"Longer than ref - effective k: {effective_k:.3f}")

    debug_print(f"Overall scale factor: {scale_factor:.3f}")

    # Determine progression type
    if course.total_km > config.MEDIUM_DISTANCE_KM:
        grace_distance_km = min(config.SHORT_DISTANCE_KM, course.total_km * 0.15)
        debug_print(f"Ultra race - using sigmoid progression with {grace_distance_km:.1f}km grace zone")

    # Apply progressively through the race
    scaled_times = []
    for i, x in enumerate(course.leg_ends_x):
        # NEW: Use different progression curves based on race length
        if course.total_km <= config.MEDIUM_DISTANCE_KM:
            # For races up to 80km, use existing power law progression
            if is_flat_road:
                progress_factor = x ** config.PROGRESS_EXP_SHORT
            elif course.total_km < config.SHORT_DISTANCE_KM:
                progress_factor = x ** config.PROGRESS_EXP_MEDIUM
            else:
                progress_factor = x ** config.PROGRESS_EXP_LONG
        else:
            # For ultra races > 80km, use sigmoid-based progression
            # This delays the onset of scaling and provides more realistic early splits

            # Define the "grace zone" - the early part of the race with minimal scaling
            # For a 250k, the first 20-30k should be relatively unaffected
            grace_distance_km = min(config.SHORT_DISTANCE_KM,
                                    course.total_km * 0.15)  # 15% of race or SHORT_DISTANCE_KM, whichever is smaller
            grace_fraction = grace_distance_km / course.total_km

            if x <= grace_fraction:
                # In the grace zone: apply minimal scaling
                # Linear ramp from 0 to 10% of the total adjustment
                progress_factor = (x / grace_fraction) * 0.1
            else:
                # After grace zone: sigmoid curve for smooth transition
                # Map remaining distance to sigmoid input [0, 6]
                remaining_x = (x - grace_fraction) / (1 - grace_fraction)
                sigmoid_input = remaining_x * 6  # Maps [0,1] to [0,6]

                # Sigmoid function: starts slow, accelerates, then plateaus
                sigmoid_value = 1 / (1 + np.exp(-sigmoid_input + 3))

                # Combine grace zone contribution with sigmoid
                # 0.1 from grace zone + 0.9 from sigmoid
                progress_factor = 0.1 + 0.9 * sigmoid_value

        # Apply the adjustment
        adjustment = 1 + (scale_factor - 1) * progress_factor

        # SAFETY: Cap the maximum slowdown for early checkpoints
        # This prevents unrealistic times like 4h for 10k
        if course.total_km > config.EXTREME_DISTANCE_KM:
            # For extreme ultras, cap early slowdown more aggressively
            distance_km = x * course.total_km
            old_adjustment = adjustment
            if distance_km <= config.SHORT_DISTANCE_KM:
                adjustment = min(adjustment, 1.10)
            elif distance_km <= config.DEFAULT_DISTANCE_KM:
                adjustment = min(adjustment, 1.20)
            elif distance_km <= config.EXTREME_DISTANCE_KM / 2:
                adjustment = min(adjustment, 1.35)

            if old_adjustment != adjustment:
                debug_print(f"  Capped adjustment at {distance_km:.1f}km: {old_adjustment:.3f} -> {adjustment:.3f}")

        scaled_times.append(base_times[i] * adjustment)

        # Debug output for key checkpoints
        dist_km = x * course.total_km
        if i < 3 or (9 <= dist_km <= 11) or i == len(course.leg_ends_x) - 1:
            debug_print(f"  Checkpoint {i}: {dist_km:.1f}km")
            debug_print(f"    Progress factor: {progress_factor:.3f}, Adjustment: {adjustment:.3f}x")
            debug_print(f"    Time: {format_time(base_times[i])} -> {format_time(scaled_times[i])}")

    debug_print(f"Scaled finish time: {format_time(scaled_times[-1])}")
    debug_print(f"Total scaling effect: {scaled_times[-1] / base_times[-1]:.3f}x")

    return np.array(scaled_times)


def apply_ultra_adjustments(times, course):
    """
    Apply ultra adjustments using config functions for cleaner code.

    Args:
        times: Array of checkpoint times
        course_km: Total race distance

    Returns:
        Tuple of (adjusted_times, metadata_dict)
    """
    finish_hours = times[-1] / config.SECONDS_PER_HOUR
    course_km = course.total_km

    debug_print("=" * 60)
    debug_print("STAGE 3: APPLY ULTRA ADJUSTMENTS")
    debug_print(f"Scaled finish time: {finish_hours:.1f} hours")

    # No adjustments for short races
    if finish_hours <= config.ULTRA_START_HOURS:
        debug_print(f"No ultra adjustments needed (< {config.ULTRA_START_HOURS} hours)")
        return times, {
            "ultra_adjusted": False,
            "slow_factor_finish": 1.0,
            "rest_added_finish_s": 0.0,
        }

    adjusted = times.copy()

    # Get fatigue and rest from config functions
    fatigue_factor = 1.0 + config.FATIGUE_SLOPE * (finish_hours - config.ULTRA_START_HOURS)  # (1.0 = no fatigue). Function derived empirically to match race time data
    total_rest_h = config.REST_SLOPE * (finish_hours - config.ULTRA_START_HOURS)  # start adding rest for courses > 5h.
    total_rest_s = total_rest_h * config.SECONDS_PER_HOUR

    debug_print(f"Base fatigue factor: {fatigue_factor:.3f}x")
    debug_print(f"Base rest time: {format_time(total_rest_s)}")

    # Apply special adjustments for ultra races
    if course_km > config.EXTREME_DISTANCE_KM:
        total_rest_s *= config.EXTREME_DISTANCE_FACTOR
        fatigue_factor *= config.EXTREME_FATIGUE_FACTOR
        debug_print(f"Extreme distance adjustments applied (>{config.EXTREME_DISTANCE_KM}km)")

    debug_print(f"Final fatigue factor at finish: {fatigue_factor:.3f}x")
    debug_print(f"Total rest time for entire race: {format_time(total_rest_s)}")

    # Apply adjustments progressively through the race
    adjusted = []

    # Calculate incremental times (time for each leg)
    incremental_times = np.diff(np.concatenate([[0], times]))

    for i in range(len(times)):
        progress = (i + 1) / len(times)
        distance_km = course.leg_ends_x[i] * course.total_km

        # FATIGUE: Builds progressively, but only affects running time
        # Use an S-curve that starts slow
        fatigue_progress = progress ** 1.5  # More gradual than linear
        current_fatigue = 1.0 + (fatigue_factor - 1.0) * fatigue_progress

        # REST: Distributed based on distance and effort
        # Minimal rest early, more rest later in the race
        if distance_km <= config.SHORT_DISTANCE_KM:
            # First part: minimal rest (maybe a quick water stop)
            rest_fraction = config.SHORT_REST_PCT
        elif distance_km <= config.DEFAULT_DISTANCE_KM:
            rest_fraction = config.MED_REST_PCT
        elif distance_km <= config.MEDIUM_DISTANCE_KM:
            # 50-100k: moderate rest
            rest_fraction = config.LONG_REST_PCT
        else:
            # Beyond 100k: rest scales with distance
            # Use a curve that gives more rest later
            rest_progress = (distance_km - 100) / (course.total_km - 100) if course.total_km > 100 else 1
            rest_fraction = config.LONG_REST_PCT + (1-config.LONG_REST_PCT) * (rest_progress ** 0.8)

        # Calculate rest time up to this checkpoint
        checkpoint_rest = total_rest_s * rest_fraction

        # Apply fatigue to the running time only, then add rest
        # Sum of all leg times with fatigue applied
        running_time = 0
        for j in range(i + 1):
            # Apply progressive fatigue to each leg
            leg_progress = (j + 1) / len(times)
            leg_fatigue = 1.0 + (fatigue_factor - 1.0) * (leg_progress ** 1.5)
            running_time += incremental_times[j] * leg_fatigue

        adjusted_time = running_time + checkpoint_rest
        adjusted.append(adjusted_time)

        # Debug output for key checkpoints
        if i < 3 or (9 <= distance_km <= 11) or i == len(times) - 1:
            debug_print(f"  Checkpoint {i}: {distance_km:.1f}km")
            debug_print(f"    Running fatigue: {current_fatigue:.3f}x")
            debug_print(f"    Rest time at this checkpoint: {format_time(checkpoint_rest)}")
            debug_print(f"    Time: {format_time(times[i])} -> {format_time(adjusted_time)}")
            debug_print(f"    Net change: +{format_time(adjusted_time - times[i])}")

    debug_print(f"Ultra-adjusted finish time: {format_time(adjusted[-1])}")
    debug_print(f"Total time added: {format_time(adjusted[-1] - times[-1])}")

    metadata = {
        "ultra_adjusted": True,
        "slow_factor_finish": float(fatigue_factor),
        "rest_added_finish_s": float(total_rest_s),
        "finish_hours": float(finish_hours),
    }

    return np.array(adjusted), metadata


def simulate_with_conditions(baseline_times, raw_base_times, course, speeds, sigmas, conditions, sims=config.MC_SIMS):
    n = len(baseline_times)
    samples = np.zeros((sims, n))

    # Per-leg scalers: how much each leg should be stretched by fatigue+rest+distance effects
    raw_incr = np.diff(np.hstack([0.0, raw_base_times]))
    base_incr = np.diff(np.hstack([0.0, baseline_times]))
    leg_scale = np.divide(base_incr, np.maximum(raw_incr, config.EPSILON))

    # Road vs trail + variance sizing (unchanged)
    gain_per_km = course.gain_m / course.total_km if course.total_km > 0 else 0
    is_road = gain_per_km < config.ROAD_GAIN_PER_KM
    race_hours = baseline_times[-1] / 3600.0
    rel_var = _get_relative_variance(race_hours, is_road)

    # tails/conditions/grade as you already do...
    base = config.TAILCAP_BASE_ROAD if is_road else config.TAILCAP_BASE_TRAIL
    k_tail = config.TAILCAP_K_ROAD if is_road else config.TAILCAP_K_TRAIL
    tail_cap = np.clip(base + k_tail * rel_var, config.TAILCAP_MIN, config.TAILCAP_MAX)
    grade_variation = config.GRADE_VAR_ROAD if is_road else config.GRADE_VAR_TRAIL
    condition_mean = -conditions * config.COND_MEAN_PER_UNIT
    condition_var = 1.0 - conditions * config.COND_VAR_PER_UNIT

    p_norm = config.PROB_NORMAL
    p_off = config.PROB_NORMAL + config.PROB_OFF
    p_good = config.OUTLIER_GOOD_SHARE

    for s in range(sims):
        u = np.random.random()
        if u < p_norm:
            day_factor = np.random.normal(0.0, config.NORMAL_SIGMA_MULT * rel_var)
        elif u < p_off:
            day_factor = abs(np.random.normal(0.0, config.OFF_SIGMA_MULT * rel_var))
        else:
            if np.random.random() < p_good:
                day_factor = -abs(np.random.normal(0.0, config.AMAZING_SIGMA_MULT * rel_var))
            else:
                mu = np.log(max(1e-12, 0.8 * rel_var))
                tail = np.random.lognormal(mean=mu, sigma=config.LOGNORM_SIGMA)
                day_factor = min(tail, tail_cap)

        total_factor = 1.0 + condition_mean + day_factor * condition_var
        grade_factors = np.maximum(np.random.normal(1.0, sigmas * grade_variation), 0.3)

        total = 0.0
        for i, leg_dists in enumerate(course.legs_meters):
            # your raw leg time from grade bins:
            varied_speeds = np.maximum(speeds * grade_factors, 0.1)
            leg_time_raw = np.sum(leg_dists / varied_speeds)

            # **apply per-leg baseline scaling** then day/conditions:
            leg_time = leg_time_raw * leg_scale[i]
            total += leg_time * total_factor
            samples[s, i] = total

    return (np.percentile(samples, 10, axis=0),
            np.percentile(samples, 50, axis=0),
            np.percentile(samples, 90, axis=0))


def run_prediction_simulation(course, pace_model, conditions=0):
    """
    Main prediction entry point with improved ultra handling.

    This is the main entry point for predictions, using:
    1. Grade-based speeds (adjusted for altitude)
    2. YOUR personal Riegel k for distance scaling
    3. Progressive application (2nd half slower)
    4. Proper ultra adjustments with metadata
    5. Single conditions parameter
    6. Monte Carlo with realistic variance

    Args:
        course: Course object with GPX data
        pace_model: PaceModel with your historical data
        conditions: Race conditions from -2 (terrible) to +2 (perfect)

    Returns:
        dict with p10, p50, p90 arrays and metadata
    """
    debug_print("\n" + "=" * 60)
    debug_print("STARTING RACE PREDICTION")
    debug_print("=" * 60)

    # Step 1: Altitude adjustment
    altitude_factor = altitude_impairment_multiplicative(course.median_altitude)
    speeds = pace_model.sea_level_speeds * altitude_factor

    debug_print(f"Altitude: {course.median_altitude:.0f}m -> speed factor: {altitude_factor:.3f}")

    # Step 2: Calculate base times from grade bins
    base_times = calculate_base_times(course, speeds)

    # Step 3: Apply YOUR personal distance scaling
    scaled_times = apply_distance_scaling(base_times, course, pace_model)

    # Step 4: Apply ultra adjustments if needed (>6 hours)
    adjusted_times, ultra_meta = apply_ultra_adjustments(scaled_times, course)

    debug_print("=" * 60)
    debug_print("STAGE 4: MONTE CARLO SIMULATION")
    debug_print(f"Conditions: {conditions} (±2 scale)")
    debug_print(f"Running {config.MC_SIMS} simulations...")

    # Step 5: Monte Carlo simulation WITH conditions
    p10, p50, p90 = simulate_with_conditions(
        adjusted_times,  # ultra-adjusted baseline (fatigue + rest)
        base_times,  # raw speed baseline (pre-scaling)
        course, speeds, pace_model.sigmas, conditions, sims=config.MC_SIMS
    )

    debug_print("\n" + "=" * 60)
    debug_print("FINAL RESULTS SUMMARY")
    debug_print("=" * 60)

    # Show key checkpoint results
    for i in range(len(course.leg_ends_x)):
        dist_km = course.leg_ends_x[i] * course.total_km
        if i < 3 or (9 <= dist_km <= 11) or i == len(course.leg_ends_x) - 1:
            debug_print(f"Checkpoint {i}: {dist_km:.1f}km")
            debug_print(f"  P10: {format_time(p10[i])}")
            debug_print(f"  P50: {format_time(p50[i])}")
            debug_print(f"  P90: {format_time(p90[i])}")

    debug_print("\nTransformation summary:")
    debug_print(f"  Base finish: {format_time(base_times[-1])}")
    debug_print(f"  After distance scaling: {format_time(scaled_times[-1])} ({scaled_times[-1] / base_times[-1]:.3f}x)")
    debug_print(
        f"  After ultra adjustments: {format_time(adjusted_times[-1])} ({adjusted_times[-1] / base_times[-1]:.3f}x)")
    debug_print(f"  Final P50: {format_time(p50[-1])}")

    # Build metadata for transparency
    k_used = get_distance_specific_k(pace_model, course.total_km)

    metadata = {
        "course_median_alt_m": float(course.median_altitude),
        "alt_speed_factor": float(altitude_factor),
        "riegel_k": float(k_used),
        "riegel_k_global": float(pace_model.riegel_k),
        "ref_distance_km": float(pace_model.ref_distance_km) if pace_model.ref_distance_km else 0.0,
        "ref_time_s": float(pace_model.ref_time_s) if pace_model.ref_time_s else None,
        "distance_scale_factor": float(scaled_times[-1] / base_times[-1]) if base_times[-1] > 0 else 1.0,
        "conditions": int(conditions),
        "finish_time_p50_s": float(p50[-1]),
        # Add ultra metadata
        **ultra_meta
    }

    return {
        "p10": p10,
        "p50": adjusted_times,
        "p90": p90,
        "metadata": metadata
    }


def _get_relative_variance(race_hours, is_road=False):
    """
    Calculate relative variance (coefficient of variation) based on race duration.

    Returns: relative standard deviation (e.g., 0.15 = 15% variance)
    """
    if is_road:
        return config.DEFAULT_VARIANCE if race_hours > 2 else config.MIN_VARIANCE
    else:
        if race_hours < 4:
            return config.DEFAULT_VARIANCE
        else:
            # Multi-day races have slightly more variance (weather, strategy)
            variance = config.MIN_VARIANCE + config.VARIANCE_EXP * np.exp(-race_hours / 12) + config.REBOUND * (
                        race_hours / (race_hours + 48))
            return min(variance, config.MAX_VARIANCE)  # Up to max variance as defined in config