# Improvement Ideas

## Performance trend chart (My Data tab)

Show how fitness has evolved over time using a Riegel-normalized, altitude-corrected equivalent time.

- Normalize all race times to a 50km equivalent using `T_norm = T_actual * (50 / D_actual)^k`
- Apply altitude correction (6%/1000m above 300m) to get sea-level equivalent
- Scatter plot of normalized time vs race date, with a trend line
- Color or size dots by actual race distance to flag outlier distances where the single-k Riegel assumption is weaker
- Data is already in `used_races.csv`; no new Strava calls needed

## Validation: CSV export and improved plots

Done (added in PR #1):
- `--csv` flag exports per-race results to `data/validation/results.csv`
- Plots now have proper legends, sample counts on box plots, summary stats annotation
- README documents what each plot panel shows

## Detect missing races

When rebuilding the model, compare the number of race-tagged activities on Strava vs races in the model and surface any that were filtered out (wrong sport_type, missing streams, etc.) so the user knows what's excluded and why.

---

## Model improvements

### Use training runs for pace curves (carefully)
Training runs are currently ignored entirely. They're noisier (easier effort, more stopping) but contain useful signal:
- **Grade-specific speed ceiling**: your fastest training segments on a given grade are closer to race effort than your average training pace. Could use e.g. 80th percentile of training speeds per grade bin as a soft upper bound, or blend training P80 with race median.
- **Volume-weighted coverage**: training runs fill in grade bins that your races don't cover well (e.g. if you race mostly flat but train on hills).
- **Recency signal**: if your most recent long training run is faster than your race from 2 years ago, the model should know.
- **Implementation**: add a `use_training` toggle. Filter to runs >5km with `workout_type != 1`. Apply a "race factor" discount (~5-10%) to training speeds before blending with race data. Weight training data lower than race data (e.g. 0.3x).

### Piecewise Riegel exponent
The single Riegel k doesn't capture how endurance degrades nonlinearly. A runner might scale well from 10k to 50k (k=1.05) but poorly from 50k to 200k (k=1.25). Consider:
- Fit separate k values for short (<30k), medium (30-80k), and long (>80k) distance ranges
- Or use a smooth function: `k(D) = k_base + k_slope * log(D/D_ref)`
- Requires enough races across distance ranges to fit

### Learn altitude sensitivity per runner
Current model uses a fixed 6% per 1000m for everyone. But altitude sensitivity varies a lot (acclimatized mountain runners vs. sea-level runners). Could learn a personal `alpha` from races at different altitudes — compare predicted vs actual for high-altitude races and adjust.

### Separate rest model for multi-day races
The rest model currently assumes single-day behavior. For races >24h (e.g. 200+ miles), sleep breaks are qualitatively different from aid station stops. Could:
- Detect sleep breaks (>30min stopped periods) separately from short rests
- Model sleep as a fixed block (e.g. 2-4h per 24h of race) rather than log-linear
- Only applies to a handful of races so may not be worth the complexity

---

## UI / UX improvements

### Race strategy input
Let users override the learned rest model per aid station. E.g. "I plan to spend 15min at AS3 and sleep 2h at AS5." This would replace the predicted rest at those checkpoints while keeping the model's estimate elsewhere. Useful for experienced ultra runners who have a plan.

### Split view: predicted vs actual
After a race, let the user input their actual splits (or pull from the Strava activity) and show predicted vs actual at each checkpoint. Would make validation tangible and help users understand where the model is strong/weak.

### GPX input validation
Warn the user on upload if:
- GPX has no elevation data (grade model won't work)
- Very few points (<100) — likely too sparse for accurate distance
- Elevation range suspiciously small for the claimed distance (possible flat file)
- Total distance differs significantly from what they might expect

### Separate condition knobs
Replace the single -2/+2 slider with independent inputs:
- Weather (temperature/wind/rain)
- Fitness / form (taper quality, recent training load)
- Course technicality (beyond what grade bins capture — scrambling, river crossings, navigation)
- Motivation / race importance

Each could shift mean and/or variance independently. More expressive but more complex UI — could be an "advanced" toggle.

---

## Deployment & accessibility

### ~~Streamlit Cloud / hosted version~~ ✅ Done
Deployed to Streamlit Community Cloud with OAuth routed through a Cloudflare Worker (holds client_secret server-side). Each user authenticates with their own Strava account.

### Docker container
Create a Dockerfile so anyone can run `docker run -p 8501:8501 race-predictor` without installing Python or dependencies. Include a `docker-compose.yml` with a persistent volume for `data/`.

### Onboarding flow
For users who don't know git:
- Landing page that explains what the tool does with example screenshots
- Guided Strava setup wizard (step-by-step with screenshots)
- "Demo mode" with a pre-built model and sample GPX so people can try it before connecting Strava
- Progressive disclosure: hide advanced settings (recency weighting, conditions) behind expandable sections

### Pre-built race courses
Ship a library of popular race GPX files (UTMB, Western States, Hardrock, etc.) with pre-configured aid stations. User just picks a race from a dropdown instead of uploading GPX + typing aid station distances. Community could contribute courses.

---

## Code quality

### Unit tests
No tests currently. Priority areas:
- `_extract_rest_data` / `_extract_fatigue_data` — edge cases with bad stream data
- `_calibrate_variance_scale` — verify z-score math
- `apply_distance_scaling` — Riegel dampening at various distances
- `apply_ultra_adjustments` — rest + fatigue interaction
- GPX parsing with malformed files

### Type hints
Add throughout — especially function signatures in `pace_builder.py` and `prediction.py`. Would catch bugs earlier and improve IDE support.

---

## Security hardening

### Validate OAuth state parameter
The `state` param is passed through the OAuth flow but not validated on callback (`app.py:handle_oauth_callback`). Should store expected state in `st.session_state` before redirecting and check it when the callback arrives, to prevent CSRF.

### Validate worker_url
`worker_url` from `st.secrets` is used directly in `requests.post()` without checking scheme or domain (`utils/strava.py`). Should validate it's HTTPS and matches `*.workers.dev` to prevent SSRF if secrets are misconfigured.
