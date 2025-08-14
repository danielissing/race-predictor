# scripts/validate_history.py
# Quick sanity check: compare predicted P50 vs actual splits on races used to build the model.

import json, os
from pathlib import Path
import numpy as np
import pandas as pd

# --- Config (tweak here) ------------------------------------------------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "strava_cache"
OUT_DIR   = DATA_DIR / "validation"
CHECKPOINT_KM = 10.0      # uniform checkpoints (e.g., 10 km); finish is auto-added
STEP_LENGTH   = 10.0      # resample step for course (meters)
STEP_WINDOW   = 40.0      # smoothing window for grade (meters)
HEAT = "moderate"         # normal conditions
FEEL = "ok"               # normal conditions
APPLY_LEAVE_ONE_OUT = False  # simple sanity = False; set True for LOO (slower; needs API utils)

# --- Imports from your utils (installed in your project) ----------------------
# We use your existing helpers so results match the app.
from utils.gpx_utils import resample_with_grade, segment_stats, altitude_impairment_multiplicative  # already in your repo
# Optional: if you want to use the exact simulator from your app:
try:
    from models.predictor import simulate_etas  # your Monte Carlo (P10/P50/P90)
    HAVE_SIM = True
except Exception:
    HAVE_SIM = False


# --- Small helpers ------------------------------------------------------------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_streams(activity_id: int):
    p = CACHE_DIR / f"streams_{activity_id}.json"
    if not p.exists():
        return None
    return load_json(p)

def build_df_from_streams(streams: dict) -> pd.DataFrame:
    """Create a dataframe with dist_m, ele_m, lat, lon from Strava streams."""
    dist = streams.get("distance", {}).get("data")
    time = streams.get("time", {}).get("data")
    alt  = streams.get("altitude", {}).get("data")
    latlng = streams.get("latlng", {}).get("data")

    if dist is None or time is None:
        raise ValueError("Missing distance/time streams")

    df = pd.DataFrame({"dist_m": pd.Series(dist, dtype="float"),
                       "t_s":    pd.Series(time, dtype="float")})
    if alt is not None and len(alt) == len(df):
        df["ele_m"] = pd.Series(alt, dtype="float")
    else:
        df["ele_m"] = 0.0

    if latlng is not None and len(latlng) == len(df):
        lat = [x[0] for x in latlng]; lon = [x[1] for x in latlng]
        df["lat"] = pd.Series(lat, dtype="float")
        df["lon"] = pd.Series(lon, dtype="float")
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan
    return df

def uniform_checkpoints_km(total_km: float, step_km: float) -> list[float]:
    pts = []
    k = step_km
    while k < total_km - 1e-6:
        pts.append(k)
        k += step_km
    pts.append(total_km)  # ensure finish
    return pts

def distance_by_grade_bins_df(df_res: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    """Meters in each grade bin for the *whole* df (assumes uniform STEP_LENGTH)."""
    grades = df_res["grade_pct"].to_numpy(dtype=float)
    # count samples per bin, multiply by step length
    idx = np.digitize(grades, bins, right=False) - 1
    idx = np.clip(idx, 0, len(bins)-2)
    counts = np.bincount(idx, minlength=len(bins)-1)
    return counts.astype(float) * STEP_LENGTH

def legs_from_checkpoints(df_res: pd.DataFrame, checkpoints_km: list[float]):
    """Return list of (start_idx, end_idx) in df_res for each leg."""
    d = df_res["dist_m"].to_numpy(dtype=float)
    legs = []
    prev_idx = 0
    for km in checkpoints_km:
        tgt = km * 1000.0
        end_idx = int(np.searchsorted(d, tgt, side="right"))
        end_idx = max(prev_idx, min(end_idx, len(d)-1))
        legs.append((prev_idx, end_idx))
        prev_idx = end_idx
    return legs

def legs_meters_by_bins(df_res: pd.DataFrame, legs_idx: list[tuple], bins: np.ndarray) -> list[np.ndarray]:
    arr = []
    for a,b in legs_idx:
        seg = df_res.iloc[a:b+1]
        arr.append(distance_by_grade_bins_df(seg, bins))
    return arr

def actual_times_at_checkpoints(dist_m: np.ndarray, t_s: np.ndarray, checkpoints_km: list[float]) -> np.ndarray:
    """Interpolate actual times (s) at cumulative distance checkpoints."""
    ck_m = np.array(checkpoints_km, dtype=float) * 1000.0
    # make sure distance is non-decreasing:
    d = np.maximum.accumulate(dist_m.astype(float))
    # ensure strictly increasing for interp
    eps = np.maximum(1e-3, np.diff(d, prepend=d[0]))
    d2 = np.cumsum(eps)
    return np.interp(ck_m, d2, t_s.astype(float))

def riegel_scale(total_km: float, p50_finish_s: float, k: float, ref_km: float | None, ref_s: float | None) -> float:
    """Return scale s >= 1 to slow down to Riegel forecast; <=1 returns 1.0 (never speed up)."""
    if not ref_km or not ref_s or total_km <= 0 or p50_finish_s <= 0:
        return 1.0
    t_riegel = float(ref_s) * (float(total_km) / float(ref_km)) ** float(k)
    s = float(t_riegel / max(1e-6, p50_finish_s))
    return s if s > 1.0 else 1.0

def apply_ultra_progressive(p10, p50, p90, leg_ends_x):
    """Same progressive ultra adjustment the app uses (no params exposed)."""
    T50 = float(p50[-1]); hours = T50 / 3600.0
    START_TH_H, ULTRA_GAMMA, REST_PER_24H = 20.0, 0.30, 2.0
    p10, p50, p90 = p10.copy(), p50.copy(), p90.copy()
    if hours <= START_TH_H:
        return p10, p50, p90
    for i, x in enumerate(leg_ends_x):
        h_here = hours * float(x)
        blocks = max(0.0, (h_here - START_TH_H) / 24.0)
        slow_i = 1.0 + ULTRA_GAMMA * blocks
        rest_s = (REST_PER_24H * 3600.0) * blocks
        p10[i] = p10[i] * slow_i + 0.25 * rest_s
        p50[i] = p50[i] * slow_i + 0.50 * rest_s
        p90[i] = p90[i] * slow_i + 1.00 * rest_s
    return p10, p50, p90

def fmt_hms(sec: float) -> str:
    sec = int(round(float(sec))); h = sec // 3600; m = (sec % 3600)//60; s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}"

# --- Load model artifacts -----------------------------------------------------
pace_path = DATA_DIR / "pace_curves.csv"
used_path = DATA_DIR / "used_races.csv"
meta_path = DATA_DIR / "model_meta.json"
assert pace_path.exists(), f"Missing {pace_path}"
assert used_path.exists(), f"Missing {used_path}"
assert meta_path.exists(), f"Missing {meta_path}"

pace_df = pd.read_csv(pace_path)
used_df = pd.read_csv(used_path)
meta    = load_json(meta_path)

if meta.get("recency_mode") != "off":
    print(f"[WARN] model_meta.json shows recency_mode='{meta.get('recency_mode')}'. "
          f"For this sanity check, rebuild curves with Recency='off' in the app.")

# Reconstruct grade bins from the pace_df
bins = np.concatenate([
    pace_df["lower_pct"].to_numpy(dtype=float),
    [float(pace_df["upper_pct"].iloc[-1])]
])

speeds_sl = pace_df["speed_mps"].to_numpy(dtype=float)
sigmas    = pace_df["sigma_rel"].to_numpy(dtype=float)

k   = float(meta.get("riegel_k", 1.06))
Dref = meta.get("ref_distance_km", None)
Tref = meta.get("ref_time_s", None)

OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
miss = 0

for _, r in used_df.iterrows():
    aid = int(r["id"])
    name = r.get("name", "")
    d_km = float(r.get("distance_km", 0.0))
    t_s  = float(r.get("elapsed_time_s", 0.0))
    streams = load_streams(aid)
    if not streams:
        miss += 1
        print(f"[SKIP] no cached streams for {aid} ({name})")
        continue

    # Build course from the *race itself*
    df = build_df_from_streams(streams)
    df_res = resample_with_grade(df, step_m=STEP_LENGTH, window_m=STEP_WINDOW)
    total_km = float(df_res["dist_m"].iloc[-1]) / 1000.0
    ck_list = uniform_checkpoints_km(total_km, CHECKPOINT_KM)
    legs_idx = legs_from_checkpoints(df_res, ck_list)
    legs_meters = legs_meters_by_bins(df_res, legs_idx, bins)
    leg_ends_x  = [(km / total_km) for km in ck_list] if total_km > 0 else []

    # Course altitude factor
    H_course = float(np.nanmedian(df_res["ele_m"].to_numpy(dtype=float)))
    A_course = altitude_impairment_multiplicative(H_course)

    # Prediction under "normal" conditions
    speeds = speeds_sl * A_course

    if HAVE_SIM:
        # Use your simulator to get P10/P50/P90; "heat/feel" are not inputs here,
        # but if your simulate_etas needs them, adjust accordingly.
        p10, p50, p90 = simulate_etas(legs_meters, speeds, sigmas, leg_ends_x, heat="moderate", feel="ok", sims=1000)
    else:
        # Deterministic P50 only (no Monte Carlo): sum meters/speed per leg
        p50 = np.array([np.sum(leg / np.maximum(speeds, 1e-6)) for leg in legs_meters], dtype=float)
        p50 = np.cumsum(p50)
        # Crude P10/P90 using sigma_rel envelope
        up = np.sum([leg * sigmas for leg in legs_meters], axis=0) if len(legs_meters) > 0 else 0.0
        p10 = p50 * 0.95
        p90 = p50 * 1.05

    # Riegel scaling (only slow down)
    s = riegel_scale(total_km, p50[-1], k, Dref, Tref)
    p10, p50, p90 = p10 * s, p50 * s, p90 * s

    # Ultra progressive adjustments
    p10, p50, p90 = apply_ultra_progressive(p10, p50, p90, leg_ends_x)

    # Actual times at the same checkpoints (interpolate on raw stream)
    t_actual = actual_times_at_checkpoints(
        dist_m=df["dist_m"].to_numpy(dtype=float),
        t_s=df["t_s"].to_numpy(dtype=float),
        checkpoints_km=ck_list
    )

    # Final metrics (finish + MAPE at checkpoints)
    finish_pred = float(p50[-1])
    finish_act  = float(t_actual[-1]) if len(t_actual) else float(t_s)
    finish_err  = (finish_pred - finish_act) / max(1e-6, finish_act)

    abs_pct_errs = np.abs((p50 - t_actual) / np.maximum(1e-6, t_actual))
    mape_ck = float(np.nanmedian(abs_pct_errs)) if len(abs_pct_errs) else np.nan

    rows.append({
        "id": aid,
        "name": name,
        "distance_km": round(total_km, 2),
        "finish_actual_s": int(finish_act),
        "finish_pred_p50_s": int(finish_pred),
        "finish_abs_pct_err": round(abs(finish_err), 4),
        "finish_bias_pct": round(finish_err, 4),
        "ck_step_km": CHECKPOINT_KM,
        "ck_median_abs_pct_err": round(mape_ck, 4),
        "course_med_alt_m": round(H_course, 1),
        "riegel_scale_applied": round(s, 3),
    })

res_df = pd.DataFrame(rows)
out_csv = OUT_DIR / "validation_results.csv"
res_df.to_csv(out_csv, index=False)

if not res_df.empty:
    print(f"\nSaved: {out_csv}")
    print(f"races analyzed: {len(res_df)} (missing streams: {miss})")
    print("\nSummary (finish):")
    print(res_df[["finish_abs_pct_err", "finish_bias_pct"]].describe(percentiles=[0.5]).round(3))
    print("\nMedian ck MAPE:", round(res_df["ck_median_abs_pct_err"].median(), 3))
else:
    print("No races analyzed (missing streams?).")
