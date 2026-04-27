import math
import pandas as pd
import numpy as np
from utils.gpx_parsing import parse_gpx, parse_cumulative_dist
from utils.elevation import resample_with_grade, segment_stats
from utils.course_analysis import legs_from_aid_stations, distance_by_grade_bins
import config

class Course:
    """
    Represents a race course, handling all GPX parsing, segmenting, and statistics.
    """

    def __init__(self, gpx_bytes: bytes, aid_km_text: str, aid_units: str):
        self.gpx_bytes = gpx_bytes
        self.aid_km_text = aid_km_text
        self.aid_units = aid_units
        self.grade_bins = config.GRADE_BINS
        self.step_length = config.STEP_LENGTH
        self.step_window = config.STEP_WINDOW
        self._compute_context()
        self._compute_fingerprint()

    def _compute_context(self):
        """
        Parses the GPX file and calculates all course-derived attributes.
        """
        self.df_raw = parse_gpx(self.gpx_bytes)
        self.df_res = resample_with_grade(self.df_raw, step_m=self.step_length, window_m=self.step_window)

        self.total_km, self.gain_m, self.loss_m, self.min_ele, self.max_ele = segment_stats(self.df_res)

        self.median_altitude = float(np.nanmedian(self.df_res['ele_m'].to_numpy(dtype=float))) if not self.df_res.empty else 0.0

        self.aid_km = parse_cumulative_dist(self.aid_km_text, self.aid_units)
        self.legs_idx = legs_from_aid_stations(self.df_res, self.aid_km)

        self.legs_meters = []
        self.leg_end_km = []
        for (a, b) in self.legs_idx:
            seg = self.df_res.iloc[a:b + 1]
            meters = distance_by_grade_bins(seg, self.grade_bins)
            if meters.sum() > 1.0:
                self.legs_meters.append(meters)
                self.leg_end_km.append(float(seg["dist_m"].iloc[-1]) / 1000.0)

        self.leg_ends_x = [min(1.0, km / max(self.total_km, config.EPSILON)) for km in self.leg_end_km]
        


class PaceModel:
    """
    Represents the runner's pacing model, built from Strava data.
    """

    def __init__(self, pace_df: pd.DataFrame, used_races_df: pd.DataFrame, meta: dict):
        self.pace_df = pace_df
        self.used_races = used_races_df
        self.meta = meta

    @property
    def riegel_k(self) -> float:
        return self.meta.get('riegel_k', config.DEFAULT_RIEGEL_K)

    @property
    def ref_distance_km(self) -> float | None:
        return self.meta.get('ref_distance_km')

    @property
    def ref_time_s(self) -> float | None:
        return self.meta.get('ref_time_s')

    @property
    def sea_level_speeds(self) -> np.ndarray:
        return self.pace_df["speed_mps"].values

    @property
    def sigmas(self) -> np.ndarray:
        return self.pace_df["sigma_rel"].values

    # --- Fatigue model ---

    @property
    def fatigue_slope(self) -> float:
        return self.meta.get("fatigue_slope", config.FATIGUE_SLOPE)

    @property
    def fatigue_n_races(self) -> int:
        return self.meta.get("fatigue_n_races", 0)

    # --- Rest model ---

    @property
    def rest_model(self) -> tuple[float, float, float]:
        """(a, b, beta) — rest_frac = a*ln(hours)+b, CDF = x^beta."""
        a = self.meta.get("rest_model_a", config.REST_FALLBACK_A)
        b = self.meta.get("rest_model_b", config.REST_FALLBACK_B)
        beta = self.meta.get("rest_distribution_beta", config.REST_FALLBACK_BETA)
        return a, b, beta

    @property
    def rest_n_races(self) -> int:
        return self.meta.get("rest_n_races", 0)

    def predict_rest_fraction(self, running_hours: float) -> float:
        """Predict rest as fraction of elapsed time, given running hours."""
        a, b, _ = self.rest_model
        if running_hours <= 0:
            return 0.0
        frac = a * math.log(running_hours) + b
        return max(0.0, min(frac, config.REST_MAX_FRACTION))

    def rest_cdf(self, progress: float) -> float:
        """Cumulative fraction of total rest taken at given race progress [0,1]."""
        _, _, beta = self.rest_model
        progress = max(0.0, min(1.0, progress))
        return progress ** beta

    # --- Variance calibration ---

    @property
    def variance_scale(self) -> float:
        return self.meta.get("variance_scale", 1.0)

    @property
    def variance_n_races(self) -> int:
        return self.meta.get("variance_n_races", 0)


class StreamCourse:
    """Build a Course-like object directly from Strava stream data.

    Reuses the same elevation resampling and grade binning pipeline
    as the GPX-based Course class, but takes distance + altitude arrays
    instead of a GPX file.

    The course is treated as a single leg (start -> finish) since for
    validation we only compare total finish time.
    """

    def __init__(self, distance_data: list, altitude_data: list):
        dist_m = np.array(distance_data, dtype=float)
        alt_m = np.array(altitude_data, dtype=float)

        # Build a DataFrame matching the format expected by resample_with_grade
        df = pd.DataFrame({"dist_m": dist_m, "ele_m": alt_m})
        df_res = resample_with_grade(df, step_m=config.STEP_LENGTH, window_m=config.STEP_WINDOW)

        # Course-level stats
        self.total_km, self.gain_m, self.loss_m, self.min_ele, self.max_ele = segment_stats(df_res)
        self.median_altitude = float(np.nanmedian(df_res["ele_m"].to_numpy(dtype=float))) if not df_res.empty else 0.0

        # Single leg: the entire course
        legs_idx = legs_from_aid_stations(df_res, [])  # no aid stations
        self.legs_meters = []
        self.leg_end_km = []
        for a, b in legs_idx:
            seg = df_res.iloc[a:b + 1]
            meters = distance_by_grade_bins(seg, config.GRADE_BINS)
            if meters.sum() > 1.0:
                self.legs_meters.append(meters)
                self.leg_end_km.append(float(seg["dist_m"].iloc[-1]) / 1000.0)

        self.leg_ends_x = [min(1.0, km / max(self.total_km, config.EPSILON)) for km in self.leg_end_km]