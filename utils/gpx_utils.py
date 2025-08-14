import numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import io, math, gpxpy
from utils.strava_utils import is_run, is_race, get_activity_streams
import hashlib

EARTH_R = 6371000.0
CLUSTER_RADIUS = 200.0
ALPHA = 0.06
H0 = 300.0
MONTH_LENGTH = 30.437
STEP_LENGTH = 10.0
STEP_WINDOW = 40.0
REST_PER_24H = 2.0  # hours of total rest per 24h of racing
ULTRA_GAMMA = 0.30  # +30% slowdown per 24h beyond threshold
START_TH_H = 20.0  # threshold (hours) before slowdowns kick in
MILD_HALF_LIFE = 18.0 # for regency considerations
DEFAULT_RIEGEL_K =  1.06 # penalty for long races
MILES_TO_KM = 1.609344
SECONDS_PER_HOUR = 3600.0

def course_fingerprint(
    gpx_bytes: bytes,
    aid_km_text: str,
    aid_units: str,
    grade_bins: list[float],
    step_length: float = 10.0,
    step_window: float = 40.0,
) -> str:
    """A deterministic key to decide when to recompute the course context."""
    h = hashlib.md5(gpx_bytes).hexdigest()
    bins_sig = ",".join(f"{x:.4f}" for x in grade_bins)
    return f"{h}|{aid_km_text.strip()}|{aid_units}|{bins_sig}|{step_length:.1f}|{step_window:.1f}"

def compute_course_context(
    gpx_bytes: bytes,
    aid_km_text: str,
    aid_units: str,
    grade_bins: list[float],
    step_length: float = 10.0,
    step_window: float = 40.0,
):
    """
    Pure function: build all course-derived artifacts, no Streamlit, no caching.
    Returns a dict with:
      df_raw, df_res, course_stats, aid_km, legs_idx, legs_meters, leg_end_km, leg_ends_x, total_km
    """
    # These functions are assumed to be in this module already:
    # parse_gpx, resample_with_grade, segment_stats, legs_from_aid_stations, distance_by_grade_bins

    df_raw = parse_gpx(gpx_bytes)
    df_res = resample_with_grade(df_raw, step_m=step_length, window_m=step_window)

    length_km, gain_m, loss_m, min_ele, max_ele = segment_stats(df_res)

    # Aid stations (to km), leg indices on the resampled track
    aid_km = parse_cumulative_dist(aid_km_text, aid_units)
    legs_idx = legs_from_aid_stations(df_res, aid_km)

    # Per-leg meters by grade bins + leg end positions
    legs_meters, leg_end_km = [], []
    total_km = float(df_res["dist_m"].iloc[-1]) / 1000.0 if len(df_res) else 0.0

    for (a, b) in legs_idx:
        seg = df_res.iloc[a:b+1]
        meters = distance_by_grade_bins(seg, grade_bins)
        if meters.sum() > 1.0:
            legs_meters.append(meters)
            leg_end_km.append(float(seg["dist_m"].iloc[-1]) / 1000.0)

    leg_ends_x = [min(1.0, km / max(total_km, 1e-6)) for km in leg_end_km] if total_km > 0 else []

    return dict(
        df_raw=df_raw,
        df_res=df_res,
        course_stats=(length_km, gain_m, loss_m, min_ele, max_ele),
        aid_km=aid_km,
        legs_idx=legs_idx,
        legs_meters=legs_meters,
        leg_end_km=leg_end_km,
        leg_ends_x=leg_ends_x,
        total_km=total_km,
    )

def _haversine_m(a1,b1,a2,b2):
    p1,p2=math.radians(a1),math.radians(a2)
    d1=math.radians(a2-a1); d2=math.radians(b2-b1)
    x=math.sin(d1/2)**2+math.cos(p1)*math.cos(p2)*math.sin(d2/2)**2
    return 2*EARTH_R*math.atan2(math.sqrt(x), math.sqrt(1-x))

def parse_cumulative_dist(text: str, units: str) -> list[float]:
    """Return a list of cumulative distances in *km* no matter what the input units are."""
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if units == "mi":
        return [v * MILES_TO_KM for v in vals]
    return vals

def parse_gpx(file_bytes, smooth_window=5):
    g=gpxpy.parse(io.StringIO(file_bytes.decode('utf-8', errors='ignore')))
    lats,lons,eles,dists=[],[],[],[]; cum=0.0; last=None
    for t in g.tracks:
        for s in t.segments:
            for p in s.points:
                if last is not None:
                    d=_haversine_m(last.latitude,last.longitude,p.latitude,p.longitude)
                    if not (d==d): d=0.0
                    cum+=d
                lats.append(p.latitude); lons.append(p.longitude)
                eles.append(p.elevation if p.elevation is not None else np.nan)
                dists.append(cum); last=p
    df=pd.DataFrame({'lat':lats,'lon':lons,'ele_m':eles,'dist_m':dists})
    df['ele_m']=df['ele_m'].interpolate().bfill().ffill()
    dd=df['dist_m'].diff().fillna(0.0).clip(lower=1e-3); de=df['ele_m'].diff().fillna(0.0)
    gr=(de/dd)*100.0
    if smooth_window>1:
        gr=gr.rolling(smooth_window,center=True,min_periods=1).median()
        gr=gr.rolling(smooth_window,center=True,min_periods=1).mean()
    df['grade_pct']=gr;
    return df

def legs_from_aid_stations(df, aid_km):
    aid_m=[k*1000.0 for k in aid_km]; idxs=[]; start=0
    for tgt in aid_m:
        end=int(np.searchsorted(df['dist_m'].values, tgt, side='right'))
        end=min(max(end,start+1), len(df)-1); idxs.append((start,end)); start=end
    if idxs and idxs[-1][1] < len(df)-1: idxs.append((idxs[-1][1], len(df)-1))
    elif not idxs: idxs.append((0, len(df)-1))
    return idxs

def distance_by_grade_bins(seg, bins):
    dd=seg['dist_m'].diff().fillna(0.0).values; grades=seg['grade_pct'].values
    bi=np.digitize(grades,bins,right=False)-1; bi=np.clip(bi,0,len(bins)-2)
    meters=np.zeros(len(bins)-1); 
    for i in range(1,len(seg)): meters[bi[i]]+=dd[i]
    return meters

def segment_stats(seg_df, resample_step_m: float = 20.0, min_step_m: float = 3.0):
    """
    Distance-based resampling + hysteresis threshold to avoid over-counting tiny wiggles.
    Returns (length_km, gain_m, loss_m, min_ele_m, max_ele_m).
    """

    d = seg_df['dist_m'].to_numpy(dtype=float)
    e = seg_df['ele_m'].to_numpy(dtype=float)
    if len(d) < 2:
        length_km = 0.0
        return length_km, 0.0, 0.0, float(e[0] if len(e) else 0), float(e[0] if len(e) else 0)

    # 1) resample elevations at fixed distance steps (e.g., 20 m)
    start, end = d[0], d[-1]
    if end - start < resample_step_m:
        d_res = np.array([start, end])
        e_res = np.interp(d_res, d, e)
    else:
        d_res = np.arange(start, end + resample_step_m, resample_step_m)
        e_res = np.interp(d_res, d, e)

    # 2) hysteresis: only count a climb/descent once it exceeds min_step_m
    gain = 0.0
    loss = 0.0
    up_acc = 0.0
    down_acc = 0.0
    prev = e_res[0]
    for x in e_res[1:]:
        delta = float(x - prev)
        prev = x
        if delta >= 0:
            up_acc += delta
            # if we had a pending descent smaller than threshold, discard it
            if down_acc <= -min_step_m:
                loss += -down_acc
            down_acc = 0.0
        else:
            down_acc += delta
            if up_acc >= min_step_m:
                gain += up_acc
            up_acc = 0.0
    # flush any final pending move
    if up_acc >= min_step_m:
        gain += up_acc
    if down_acc <= -min_step_m:
        loss += -down_acc

    length_km = float((d_res[-1] - d_res[0]) / 1000.0)
    min_ele = float(np.nanmin(e_res))
    max_ele = float(np.nanmax(e_res))
    return length_km, float(gain), float(loss), min_ele, max_ele

# --- Map helpers ---
def _haversine_m_internal(a1, b1, a2, b2):
    # Reuse the same formula as _haversine_m, but keep a public helper name
    p1, p2 = math.radians(a1), math.radians(a2)
    d1 = math.radians(a2 - a1)
    d2 = math.radians(b2 - b1)
    x = math.sin(d1 / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d2 / 2) ** 2
    return 2 * EARTH_R * math.atan2(math.sqrt(x), math.sqrt(1 - x))

def aid_station_markers(df, aid_km, cluster_radius_m: float = CLUSTER_RADIUS):
    """
    From GPX dataframe + cumulative aid station km, return clustered markers:
      [{'lat':..., 'lon':..., 'labels':['AS1','AS4'], 'kms':[10.0,42.2]}, ...]
    Stations within cluster_radius_m are merged so repeats share a marker.
    """
    pts = []
    # build raw points
    for i, km in enumerate(aid_km, start=1):
        tgt_m = km * 1000.0
        idx = int(np.searchsorted(df['dist_m'].values, tgt_m, side='right'))
        idx = max(0, min(idx, len(df) - 1))
        lat = float(df['lat'].iloc[idx]); lon = float(df['lon'].iloc[idx])
        pts.append({'lat': lat, 'lon': lon, 'label': f"AS{i}", 'km': float(km)})

    # cluster by proximity
    clusters = []
    for p in pts:
        placed = False
        for c in clusters:
            d = _haversine_m_internal(p['lat'], p['lon'], c['lat'], c['lon'])
            if d <= cluster_radius_m:
                c['labels'].append(p['label'])
                c['kms'].append(p['km'])
                placed = True
                break
        if not placed:
            clusters.append({'lat': p['lat'], 'lon': p['lon'], 'labels': [p['label']], 'kms': [p['km']]})
    return clusters

def resample_by_distance(df, step_m: float = STEP_LENGTH):
    d = df["dist_m"].to_numpy(dtype=float)
    e = df["ele_m"].to_numpy(dtype=float)
    if len(d) < 2:
        return df.copy()
    d0, d1 = float(d[0]), float(d[-1])
    grid = np.arange(d0, d1 + step_m, step_m)
    e_i = np.interp(grid, d, e)
    out = pd.DataFrame({"dist_m": grid, "ele_m": e_i})
    return out

def add_grade_over_window(df, window_m: float = STEP_WINDOW, step_m: float = STEP_LENGTH):
    """
    Smooth elevation over ~window_m and compute grade (%) with np.gradient.
    This returns a grade array with the SAME length as df (fixing the off-by-one).
    """
    import numpy as np, pandas as pd

    e = df["ele_m"].to_numpy(dtype=float)

    # Smooth: rolling median then mean over ~window_m
    w = max(3, int(round(window_m / step_m)))
    if w % 2 == 0:
        w += 1
    es = pd.Series(e).rolling(w, center=True, min_periods=1).median()
    es = es.rolling(w, center=True, min_periods=1).mean().to_numpy()

    # Grade = de/dx * 100 (%), dx = step_m
    grade = np.gradient(es, step_m) * 100.0

    out = df.copy()
    out["grade_pct"] = grade
    return out

def resample_with_grade(df, step_m: float = STEP_LENGTH, window_m: float = STEP_WINDOW):
    df2 = resample_by_distance(df, step_m=step_m)
    return add_grade_over_window(df2, window_m=window_m, step_m=step_m)


# --- Ultra adjustments: sleep + extra slowdown for very long races ---
def apply_ultra_adjustments_progressive(p10, p50, p90, leg_ends_x):
    """
    Progressive ultra adjustments:
      - multiplicative slowdown starts after START_TH_H hours and grows with race progress
      - rest time is added cumulatively after threshold (none at the start)
      - more skew to the right tail (p90)
    """

    T50 = float(p50[-1])
    hours = T50 / SECONDS_PER_HOUR

    p10 = p10.copy(); p50 = p50.copy(); p90 = p90.copy()

    if hours <= START_TH_H:
        meta = dict(
            start_threshold_h=START_TH_H,
            ultra_gamma=ULTRA_GAMMA,
            rest_per_24h_h=REST_PER_24H,
            slow_factor_finish=1.0,
            rest_added_finish_s=0.0,
        )
        return p10, p50, p90, meta

    slow_finish = 1.0
    rest_finish_s = 0.0

    for i, x in enumerate(leg_ends_x):
        # x is fraction of total distance at this checkpoint (0..1)
        h_here = hours * float(x)
        blocks = max(0.0, (h_here - START_TH_H) / 24.0)

        slow_i = 1.0 + ULTRA_GAMMA * blocks
        rest_s = (REST_PER_24H * SECONDS_PER_HOUR) * blocks

        p10[i] = p10[i] * slow_i + 0.25 * rest_s
        p50[i] = p50[i] * slow_i + 0.50 * rest_s
        p90[i] = p90[i] * slow_i + 1.00 * rest_s

        slow_finish = slow_i
        rest_finish_s = rest_s

    if hours >= 12.0:
        skew = 1.0 + 0.1 * max(0.0, (hours - 12.0) / 12.0)  # up to +6% by ~36h
        p90 = p90 * skew

    meta = dict(
        start_threshold_h=START_TH_H,
        ultra_gamma=ULTRA_GAMMA,
        rest_per_24h_h=REST_PER_24H,
        slow_factor_finish=float(slow_finish),
        rest_added_finish_s=float(rest_finish_s),
    )
    return p10, p50, p90, meta

def altitude_impairment_multiplicative(H_m: float, alpha: float = ALPHA, h0: float = H0) -> float:
    """
    Multiplicative speed factor in [0.6, 1.0].
    1.0 at/below 300 m; minus ~6% per 1000 m above 300 m (clipped at 40% total).
    """
    if H_m <= h0:
        return 1.0
    loss = alpha * ((H_m - h0) / 1000.0)
    return float(np.clip(1.0 - loss, 0.6, 1.0))

def parse_iso8601_utc(s: str) -> datetime:
    # Strava start_date is ISO8601 with 'Z'
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def recency_weight(start_date_str: str, mode: str = "mild") -> float:
    """
    Weighted by recency:
      - "off": 1.0
      - "mild": half-life ~18 months
      - "medium": half-life ~9 months
    """
    if mode == "off":
        return 1.0
    now = datetime.now(timezone.utc)
    dt = parse_iso8601_utc(start_date_str)
    months = max(0.0, (now - dt).days / MONTH_LENGTH)
    half_life = MILD_HALF_LIFE if mode == "mild" else MILD_HALF_LIFE/2
    lam = np.log(2) / half_life  # per month
    return float(np.exp(-lam * months))

def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted percentile of x at q in [0,100]."""
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]; w = w[order]
    c = np.cumsum(w)
    if c[-1] == 0:
        return float(np.median(x))
    p = c / c[-1]
    return float(np.interp(q/100.0, p, x))

def build_pace_curves_from_races(
    access_token: str,
    activities: List[Dict[str, Any]],
    bins: list,
    max_activities: int = 200,
    recency_mode: str = "mild",   # "off" | "mild" | "medium"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Build a *sea-level* pace curve from Strava RACE activities with:
      • altitude normalization (remove high-alt penalty, then re-apply at prediction),
      • recency weighting (gentle),
      • personal Riegel exponent k (for stamina scaling on very short/long races).
    Returns: (curves_df, used_races_df, meta)
      curves_df columns: lower_pct, upper_pct, speed_mps (SEA-LEVEL), sigma_rel
      used_races_df columns: id,name,date,distance_km,elapsed_time_s,median_alt_m,weight
      meta: dict(alpha, recency_mode, riegel_k, ref_distance_km, ref_time_s, n_races)
    """
    n_bins = len(bins) - 1
    # Collect weighted, sea-level normalized samples per bin
    spd_bins = [ [] for _ in range(n_bins) ]   # speeds (sea-level)
    w_bins   = [ [] for _ in range(n_bins) ]   # sample weights

    races = [a for a in activities if is_run(a) and is_race(a)]
    # de-dup by id
    seen, races_d = set(), []
    for a in races:
        aid = a.get("id")
        if aid in seen: continue
        seen.add(aid); races_d.append(a)
    races = races_d

    used_meta = []

    for a in races[:max_activities]:
        aid = a.get("id")
        streams = get_activity_streams(access_token, aid)
        if not streams:
            continue  # 404/no streams

        dist   = streams.get("distance", {}).get("data")
        vs     = streams.get("velocity_smooth", {}).get("data")
        time_s = streams.get("time", {}).get("data")
        grd    = streams.get("grade_smooth", {}).get("data")
        alt    = streams.get("altitude", {}).get("data")
        moving = streams.get("moving", {}).get("data")

        if dist is None or len(dist) < 2:
            continue

        dist_m = np.array(dist, dtype=float)

        # Speed (prefer vs; else derive from distance/time)
        if vs is not None:
            vel = np.array(vs, dtype=float)
        elif time_s is not None:
            t = np.array(time_s, dtype=float)
            dt = np.diff(t, prepend=t[0]); dd = np.diff(dist_m, prepend=dist_m[0])
            dt = np.clip(dt, 1e-3, None)
            vel = dd / dt
        else:
            continue

        # Moving fallback
        if moving is None:
            moving = (vel > 0.2).astype(int).tolist()
        mov = np.array(moving, dtype=float)

        # Grade
        if grd is None:
            if alt is not None and len(alt) == len(dist_m):
                alt_m = np.array(alt, dtype=float)
                dd = np.diff(dist_m, prepend=dist_m[0]); da = np.diff(alt_m, prepend=alt_m[0])
                dd = np.clip(dd, 1e-3, None)
                grd_arr = (da / dd) * 100.0
            else:
                grd_arr = np.zeros_like(dist_m)
        else:
            grd_arr = np.array(grd, dtype=float)

        # Median altitude for the activity (use robust median)
        median_alt = float(np.nanmedian(alt)) if alt is not None and len(alt) else 0.0
        # Sea-level normalization factor (divide observed speed by impairment)
        A_r = altitude_impairment_multiplicative(median_alt)

        # Sample weights: distance increments * movement * recency
        dd = np.diff(dist_m, prepend=dist_m[0])
        mask = (vel > 0.2) & (mov > 0.5) & (dd > 0)
        dd = dd * mask

        # Recency weight per race
        w_r = recency_weight(a.get("start_date", ""), recency_mode)

        # Bin by grade; push normalized speeds + weighted meters
        bin_idx = np.digitize(grd_arr, bins, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        used_any = False
        for i in range(1, len(dist_m)):
            if dd[i] <= 0:
                continue
            b = int(bin_idx[i])
            # Sea-level normalized speed
            s_sl = float(np.clip(vel[i] / max(A_r, 1e-6), 0.1, 6.0))
            spd_bins[b].append(s_sl)
            w_bins[b].append(float(dd[i] * w_r))
            used_any = True

        # Record meta if we actually used anything
        if used_any:
            km = (a.get("distance", 0.0) or 0.0) / 1000.0
            used_meta.append({
                "id": aid,
                "name": a.get("name", "(unnamed)"),
                "date": (a.get("start_date","") or "")[:10],
                "distance_km": round(km, 2),
                "elapsed_time_s": int(a.get("elapsed_time") or 0),
                "median_alt_m": median_alt,
                "weight": round(w_r, 3),
            })

    used_df = pd.DataFrame(used_meta)
    if not used_df.empty:
        used_df = (used_df
            .drop_duplicates(subset="id")
            .sort_values("date", ascending=False)
            .reset_index(drop=True))

    # Aggregate sea-level curve with weighted percentiles
    rows = []
    for i in range(n_bins):
        x = np.array(spd_bins[i], dtype=float)
        w = np.array(w_bins[i], dtype=float)
        if len(x) == 0 or np.sum(w) == 0:
            rows.append(dict(lower_pct=bins[i], upper_pct=bins[i+1], speed_mps=1.2, sigma_rel=0.10))
        else:
            med = weighted_percentile(x, w, 50)
            q10 = weighted_percentile(x, w, 10)
            q90 = weighted_percentile(x, w, 90)
            rel = (q90 - q10) / max(1e-6, 2 * med)
            rows.append(dict(
                lower_pct=bins[i],
                upper_pct=bins[i+1],
                speed_mps=float(med),          # SEA-LEVEL speed
                sigma_rel=float(np.clip(rel, 0.05, 0.20))
            ))
    curves = pd.DataFrame(rows)

    # --- Fit personal Riegel exponent k (log T ~ a + k log D) with recency weights ---
    k = DEFAULT_RIEGEL_K  # default if insufficient data
    ref_distance_km = None
    ref_time_s = None
    if not used_df.empty:
        # Only use races with valid (D,T)
        d = used_df["distance_km"].to_numpy(dtype=float)
        t = used_df["elapsed_time_s"].to_numpy(dtype=float)
        w = used_df["weight"].to_numpy(dtype=float)
        m = (d > 0) & (t > 0)
        if np.sum(m) >= 2:
            X = np.log(d[m]); Y = np.log(t[m]); W = w[m]
            # weighted linear regression slope (k) and intercept (a)
            WX = W * X
            WY = W * Y
            S = np.sum(W)
            SX = np.sum(WX); SY = np.sum(WY)
            SXX = np.sum(W * X * X); SXY = np.sum(W * X * Y)
            denom = (S * SXX - SX * SX)
            if denom > 0:
                k = float((S * SXY - SX * SY) / denom)

            # choose a reference race near weighted median distance
            order = np.argsort(d[m])
            d_sorted = d[m][order]; t_sorted = t[m][order]; w_sorted = W[order]
            cw = np.cumsum(w_sorted); cw = cw / cw[-1]
            idx = int(np.searchsorted(cw, 0.5))
            ref_distance_km = float(d_sorted[min(idx, len(d_sorted)-1)])
            ref_time_s = float(t_sorted[min(idx, len(t_sorted)-1)])

    meta = dict(
        alpha=ALPHA,
        recency_mode=recency_mode,
        riegel_k=float(k),
        ref_distance_km=float(ref_distance_km) if ref_distance_km else None,
        ref_time_s=float(ref_time_s) if ref_time_s else None,
        n_races=int(len(used_df)) if used_df is not None else 0,
    )
    return curves, used_df, meta