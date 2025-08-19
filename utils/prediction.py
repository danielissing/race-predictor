"""
Core prediction logic separated from UI concerns.
"""

import numpy as np
import hashlib
from utils.performance import altitude_impairment_multiplicative, apply_ultra_adjustments_progressive
from utils.simulation import simulate_etas
import config


def run_prediction_simulation(course, pace_model, feel, heat):
    """
    Run the core prediction simulation logic.

    Args:
        course: Course object with GPX data
        pace_model: PaceModel object with speed curves
        feel: User's feeling ("good", "ok", "meh")
        heat: Weather condition ("cool", "moderate", "hot")

    Returns:
        dict: Dictionary containing:
            - p10, p50, p90: Percentile predictions
            - metadata: Prediction metadata including altitude factors, Riegel scaling, etc.
    """
    # Altitude impairment
    A_course = altitude_impairment_multiplicative(course.median_altitude)
    speeds = pace_model.sea_level_speeds * A_course

    # Deterministic seed for reproducible results
    seed_payload = str([
        config.GRADE_BINS,
        list(np.round(speeds, 5)),
        list(np.round(pace_model.sigmas, 5)),
        heat, feel,
        list(np.round(course.leg_end_km, 3)),
        round(course.total_km, 3)
    ])
    seed = int(hashlib.sha256(seed_payload.encode()).hexdigest(), 16) % (2 ** 32)
    np.random.seed(seed)

    # Run simulations
    p10, p50, p90 = simulate_etas(
        course.legs_meters, speeds, pace_model.sigmas,
        course.leg_ends_x, heat, feel, sims=config.MC_SIMS
    )

    # Riegel Scaling (only slows down, never speeds up)
    T_base50 = p50[-1]
    riegel_applied = False
    riegel_scale = 1.0

    if pace_model.ref_distance_km and pace_model.ref_time_s:
        T_riegel = pace_model.ref_time_s * (course.total_km / pace_model.ref_distance_km) ** pace_model.riegel_k
        s = T_riegel / max(T_base50, config.EPSILON)
        if s > 1.0:
            p10, p50, p90 = p10 * s, p50 * s, p90 * s
            riegel_applied = True
            riegel_scale = s

    # Ultra adjustments for very long races
    p10, p50, p90, ultra_meta = apply_ultra_adjustments_progressive(p10, p50, p90, course.leg_ends_x)

    # Build metadata
    metadata = {
        "course_median_alt_m": float(course.median_altitude),
        "alt_speed_factor": float(A_course),
        "recency_mode": pace_model.meta.get("recency_mode", "mild"),
        "riegel_k": float(pace_model.riegel_k),
        "riegel_applied": bool(riegel_applied),
        "riegel_scale_factor": float(riegel_scale),
        "ref_distance_km": float(pace_model.ref_distance_km) if pace_model.ref_distance_km else None,
        "ref_time_s": float(pace_model.ref_time_s) if pace_model.ref_time_s else None,
        "finish_time_p50_s": float(p50[-1]),
        **ultra_meta,
    }

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "metadata": metadata
    }