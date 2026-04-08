#!/usr/bin/env python3
"""
Validation script for the race prediction algorithm.

Compares predictions to actual race results using real terrain data
from cached Strava streams. Two modes:
  - Quick: use existing model from disk (default)
  - LOOCV: leave-one-out cross-validation (rebuild model excluding each race)

Usage:
    python validate.py              # quick mode
    python validate.py --loocv      # leave-one-out cross-validation
    python validate.py --plot       # quick mode + save plots
    python validate.py --loocv --plot
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from models import PaceModel
from utils.course_analysis import distance_by_grade_bins, legs_from_aid_stations
from utils.elevation import resample_with_grade, segment_stats
from utils.pace_builder import build_pace_curves_from_races
from utils.persistence import load_pace_model_from_disk
from utils.prediction import run_prediction_simulation
from utils.strava import is_race, is_run

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StreamCourse — lightweight Course-like object built from cached stream data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_streams(race_id) -> dict | None:
    """Load cached streams JSON. Returns None if missing or empty."""
    path = os.path.join(config.CACHE_DIR, f"streams_{race_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None


def build_activity_dicts(races_df: pd.DataFrame) -> list[dict]:
    """Convert used_races.csv rows into activity dicts for build_pace_curves_from_races."""
    activities = []
    for _, row in races_df.iterrows():
        activities.append({
            "id": int(row["id"]),
            "name": row.get("name", ""),
            "start_date": str(row.get("date", "")),
            "distance": float(row["distance_km"]) * 1000.0,
            "elapsed_time": int(row["elapsed_time_s"]),
            "sport_type": "TrailRun",
            "workout_type": 1,  # marks as race
        })
    return activities


def format_hhmmss(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    s = int(round(seconds))
    h, rem = divmod(abs(s), 3600)
    m, sec = divmod(rem, 60)
    sign = "-" if s < 0 else ""
    return f"{sign}{h}:{m:02d}:{sec:02d}"


def _safe(text: str) -> str:
    """Strip characters that the console encoding cannot handle."""
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc)


# ---------------------------------------------------------------------------
# Single-race validation
# ---------------------------------------------------------------------------

def validate_single_race(race_row, pace_model: PaceModel) -> dict | None:
    """Validate prediction for one race using cached stream data.

    Returns a result dict or None if the race must be skipped.
    """
    race_id = str(race_row["id"])
    streams = load_streams(race_id)
    if streams is None:
        log.warning("No cached streams for race %s — skipping", race_id)
        return None

    dist_data = streams.get("distance", {}).get("data")
    alt_data = streams.get("altitude", {}).get("data")
    if not dist_data or not alt_data or len(dist_data) < 10:
        log.warning("Insufficient stream data for race %s — skipping", race_id)
        return None

    try:
        course = StreamCourse(dist_data, alt_data)
    except Exception as exc:
        log.warning("Failed to build course for race %s: %s", race_id, exc)
        return None

    if not course.legs_meters:
        log.warning("Empty legs for race %s — skipping", race_id)
        return None

    try:
        prediction = run_prediction_simulation(course, pace_model, conditions=0)
    except Exception as exc:
        log.warning("Prediction failed for race %s: %s", race_id, exc)
        return None

    actual = float(race_row["elapsed_time_s"])
    p10 = float(prediction["p10"][-1])
    p50 = float(prediction["p50"][-1])
    p90 = float(prediction["p90"][-1])

    error_pct = (p50 - actual) / actual * 100.0
    within = p10 <= actual <= p90

    return {
        "race_id": race_id,
        "name": race_row["name"],
        "date": race_row.get("date", ""),
        "distance_km": float(race_row["distance_km"]),
        "actual_s": actual,
        "p10_s": p10,
        "p50_s": p50,
        "p90_s": p90,
        "error_pct": error_pct,
        "within_ci": within,
        "gain_m": course.gain_m,
        "median_alt": course.median_altitude,
    }


# ---------------------------------------------------------------------------
# Quick validation (existing model)
# ---------------------------------------------------------------------------

def run_quick_validation(races_df: pd.DataFrame) -> list[dict]:
    pace_model = load_pace_model_from_disk()
    if pace_model is None:
        print("ERROR: No saved pace model found. Run the app first to build your model.")
        sys.exit(1)

    print(f"Loaded pace model ({len(pace_model.used_races)} races, Riegel k={pace_model.riegel_k:.3f})")
    results = []
    n = len(races_df)
    for i, (_, row) in enumerate(races_df.iterrows()):
        print(f"  [{i+1}/{n}] {_safe(row['name'][:40]):<40s} ", end="", flush=True)
        r = validate_single_race(row, pace_model)
        if r:
            results.append(r)
            flag = "OK" if r["within_ci"] else "MISS"
            print(f"err={r['error_pct']:+5.1f}%  [{flag}]")
        else:
            print("SKIP")
    return results


# ---------------------------------------------------------------------------
# LOOCV validation
# ---------------------------------------------------------------------------

def run_loocv_validation(races_df: pd.DataFrame) -> list[dict]:
    all_activities = build_activity_dicts(races_df)
    results = []
    n = len(races_df)

    for i, (_, row) in enumerate(races_df.iterrows()):
        race_id = int(row["id"])
        print(f"  [{i+1}/{n}] {_safe(row['name'][:40]):<40s} ", end="", flush=True)

        # Build model excluding this race
        loo_activities = [a for a in all_activities if a["id"] != race_id]
        if len(loo_activities) < 2:
            print("SKIP (too few remaining races)")
            continue

        try:
            curves_df, used_races_df, meta = build_pace_curves_from_races(
                access_token="dummy",  # streams are cached, token unused
                activities=loo_activities,
                bins=config.GRADE_BINS,
            )
            pace_model = PaceModel(curves_df, used_races_df, meta)
        except Exception as exc:
            print(f"SKIP (model build failed: {exc})")
            continue

        r = validate_single_race(row, pace_model)
        if r:
            results.append(r)
            flag = "OK" if r["within_ci"] else "MISS"
            print(f"err={r['error_pct']:+5.1f}%  [{flag}]")
        else:
            print("SKIP")

    return results


# ---------------------------------------------------------------------------
# Output: formatted table + summary stats
# ---------------------------------------------------------------------------

def print_results_table(results: list[dict]):
    if not results:
        print("\nNo successful validations.")
        return

    # Header
    print()
    print(f"{'Race':<35s} {'Dist':>6s} {'Actual':>8s} {'P50':>8s} {'Err%':>6s} {'P10-P90':>15s} {'CI':>4s}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x["distance_km"]):
        name = _safe(r["name"][:34])
        ci_flag = "OK" if r["within_ci"] else "MISS"
        print(
            f"{name:<35s} "
            f"{r['distance_km']:5.1f}k "
            f"{format_hhmmss(r['actual_s']):>8s} "
            f"{format_hhmmss(r['p50_s']):>8s} "
            f"{r['error_pct']:+5.1f}% "
            f"{format_hhmmss(r['p10_s']):>7s}-{format_hhmmss(r['p90_s']):<7s} "
            f"{ci_flag:>4s}"
        )

    # Summary stats
    errors = np.array([r["error_pct"] for r in results])
    abs_errors = np.abs(errors)
    coverage = sum(1 for r in results if r["within_ci"]) / len(results) * 100

    print("-" * 90)
    print(f"  N={len(results)}  "
          f"MAE={abs_errors.mean():.1f}%  "
          f"Bias={errors.mean():+.1f}%  "
          f"RMSE={np.sqrt((errors**2).mean()):.1f}%  "
          f"Coverage(P10-P90)={coverage:.0f}%")
    print()


# ---------------------------------------------------------------------------
# Plots (optional)
# ---------------------------------------------------------------------------

def save_plots(results: list[dict], output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Race Prediction Validation", fontsize=14, fontweight="bold")

    # 1) Error histogram
    ax = axes[0, 0]
    ax.hist(df["error_pct"], bins=15, alpha=0.7, edgecolor="black", color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.axvline(df["error_pct"].mean(), color="orange", ls="-", lw=1.5, label=f"mean={df['error_pct'].mean():.1f}%")
    ax.set_xlabel("Prediction error (%)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution (P50)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Actual vs predicted scatter
    ax = axes[0, 1]
    colors = ["green" if w else "red" for w in df["within_ci"]]
    ax.scatter(df["actual_s"] / 3600, df["p50_s"] / 3600, c=colors, alpha=0.7, s=50)
    lims = [
        min((df["actual_s"] / 3600).min(), (df["p50_s"] / 3600).min()) * 0.9,
        max((df["actual_s"] / 3600).max(), (df["p50_s"] / 3600).max()) * 1.1,
    ]
    ax.plot(lims, lims, "r--", lw=1.5, label="perfect")
    ax.set_xlabel("Actual (hours)")
    ax.set_ylabel("Predicted P50 (hours)")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) Error by distance (box)
    ax = axes[1, 0]
    bins_edges = [0, 30, 60, 100, 300]
    labels = ["<30k", "30-60k", "60-100k", "100k+"]
    df["dist_bin"] = pd.cut(df["distance_km"], bins=bins_edges, labels=labels)
    box_data = [df.loc[df["dist_bin"] == lab, "error_pct"].dropna().values for lab in labels]
    box_data = [d for d in box_data if len(d) > 0]
    box_labels = [lab for lab, d in zip(labels, [df.loc[df["dist_bin"] == lab, "error_pct"].dropna() for lab in labels]) if len(d) > 0]
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels)
    ax.axhline(0, color="red", ls="--", alpha=0.7)
    ax.set_ylabel("Error (%)")
    ax.set_title("Error by distance")
    ax.grid(alpha=0.3)

    # 4) Residuals vs distance
    ax = axes[1, 1]
    ax.scatter(df["distance_km"], df["error_pct"], alpha=0.6, s=40, color="steelblue")
    ax.axhline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Error (%)")
    ax.set_title("Residuals vs distance")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "validation_plots.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate race predictions against actual results.")
    parser.add_argument("--loocv", action="store_true", help="Leave-one-out cross-validation (slower, more honest)")
    parser.add_argument("--plot", action="store_true", help="Save validation plots to data/validation/")
    args = parser.parse_args()

    # Load race history
    if not os.path.exists(config.USED_RACES_PATH):
        print(f"ERROR: {config.USED_RACES_PATH} not found. Build your model first.")
        sys.exit(1)

    races_df = pd.read_csv(config.USED_RACES_PATH)
    print(f"Found {len(races_df)} races in {config.USED_RACES_PATH}")

    if args.loocv:
        print("\nRunning LOOCV validation (this may take a few minutes)...\n")
        results = run_loocv_validation(races_df)
    else:
        print("\nRunning quick validation (existing model)...\n")
        results = run_quick_validation(races_df)

    if not results:
        print("\nNo races could be validated. Check that streams are cached in", config.CACHE_DIR)
        sys.exit(1)

    print_results_table(results)

    if args.plot:
        save_plots(results, "data/validation")


if __name__ == "__main__":
    main()
