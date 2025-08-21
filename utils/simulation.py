"""
Monte Carlo race simulation functions.
Simulates race times with variability and uncertainty.
"""

import numpy as np
from utils.performance import fatigue_multiplier
import config


def simulate_confidence_intervals(
        legs_meters: np.ndarray,
        speeds_mps: np.ndarray,
        sigmas_rel: np.ndarray,
        leg_ends_x: np.ndarray,
        heat: str,
        feel: str,
        sims: int = config.MC_SIMS,
        sigma_day: float = config.SIGMA_DAY,
        rho: float = config.RHO
) -> tuple:
    """
    Run Monte Carlo simulation to predict race finish times with uncertainty.

    This is the core prediction engine that accounts for:
    1. Day-to-day performance variability
    2. Grade-specific pace uncertainty
    3. Weather and feeling adjustments
    4. Progressive fatigue throughout the race

    Args:
        legs_meters: Distance covered in each grade bin for each leg (2D array)
                    Shape: (n_legs, n_grade_bins)
        speeds_mps: Base speeds for each grade bin (meters per second)
        sigmas_rel: Relative uncertainty for each grade bin (coefficient of variation)
        leg_ends_x: Fractional progress at each checkpoint (0.0 to 1.0)
        heat: Weather condition ("cool", "moderate", "hot")
        feel: How runner feels ("good", "ok", "meh")
        sims: Number of Monte Carlo simulations to run
        sigma_day: Day-to-day performance variability (e.g., 0.05 = 5%)
        rho: Correlation between grade bins (0 to 1)

    Returns:
        Tuple of (p10, p90) - 10th and 90th percentile times for each checkpoint

    Simulation Process:
    1. Each simulation draws a "day factor" (good day vs bad day)
    2. Grade-specific speeds are adjusted with correlated noise
    3. Each leg's time is calculated considering:
       - Distance at each grade
       - Adjusted speed for each grade
       - Progressive fatigue
       - Weather/feeling multipliers
    4. Percentiles are extracted from all simulations
    """
    n_legs = len(legs_meters)
    samples = np.zeros((sims, n_legs))

    for sim in range(sims):
        # Day-to-day variability (affects entire race)
        day_factor = np.random.normal(0.0, sigma_day)

        # Grade-specific speed adjustments (correlated across grades)
        z_independent = np.random.normal(0.0, 1.0, size=speeds_mps.shape[0])

        # Create correlated noise using rho
        # eps = rho * day_component + sqrt(1-rho²) * independent_component
        eps = rho * (day_factor / max(sigma_day, 1e-9)) + np.sqrt(max(0.0, 1 - rho ** 2)) * z_independent

        # Convert to lognormal multipliers (ensures speeds stay positive)
        log_std = np.sqrt(np.log(1 + sigmas_rel ** 2))
        speed_multipliers = np.exp(-0.5 * log_std ** 2 + log_std * eps)

        # Adjusted speeds for this simulation
        adjusted_speeds = speeds_mps * speed_multipliers

        cumulative_time = 0.0

        for leg_idx in range(n_legs):
            # Calculate base time for this leg
            # Time = sum(distance_in_grade / speed_for_grade) for all grades
            base_time = np.sum(
                np.where(adjusted_speeds > 0,
                         legs_meters[leg_idx] / adjusted_speeds,
                         0.0)
            )

            # Apply all multipliers
            leg_time = (base_time *
                        fatigue_multiplier(leg_ends_x[leg_idx]) *  # Progressive fatigue
                        config.HEAT_MULT.get(heat, 1.0) *  # Weather adjustment
                        config.FEEL_MULT.get(feel, 1.0) *  # Feeling adjustment
                        (1.0 + day_factor))  # Day variability

            cumulative_time += leg_time
            samples[sim, leg_idx] = cumulative_time

    # Extract percentiles from all simulations
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)

    return p10, p90