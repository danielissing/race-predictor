import pandas as pd
import numpy as np
import hashlib
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
        
    def _compute_fingerprint(self):
        """
        A deterministic key to decide when to recompute the course context.
        """
        h = hashlib.md5(self.gpx_bytes).hexdigest()
        bins_sig = ",".join(f"{x:.4f}" for x in self.grade_bins)
        return f"{h}|{self.aid_km_text.strip()}|{self.aid_units}|{bins_sig}|{self.step_length:.1f}|{self.step_window:.1f}"
        


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