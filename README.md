# Trail Running ETA Predictor

A local Streamlit app that predicts **arrival times plus 10th/90th percentile** at each aid station for a trail race.
It uses your **Strava race history** to learn a personal **speed vs. grade** curve, combines it with a **GPX** of the course (including elevation), and includes a simple simulation to account for fatigue and conditions. There is also some helpful visualization for each leg of the race.

---

## Features

* **Strava integration (OAuth)** to pull your race activities & streams
* Personal **pace curves by grade bins** (e.g., <-20%, -12–-8%, 0–3%, 20%+) built from your past races. 
* **GPX upload** (course profile); **aid stations** as cumulative distances in **km or mi**
* **Course map** (OpenStreetMap) with your route and **aid-station markers** 
* **ETA predictions** at each aid station with **P10/P50/P90** bands 
* **Segments overview**: elevation mini-plot per leg + length (km), gain/loss (m), min/max elevation
* **Used races** table: name, date, distance—so you can sanity-check inputs
* Local caching of **Strava tokens, activity streams, pace curves** to keep API calls minimal and survive restarts

---

## Requirements

* Python 3.10+
* OS: Windows/macOS/Linux
* Strava account

See [list of Python packages](https://github.com/danielissing/race-predictor/blob/main/requirements.txt) for required libraries (installed via `requirements.txt`).

---

## Quick Start

### 1) Clone & create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate
py -m pip install -r requirements.txt
```

### 2) Run the app

```bash
streamlit run app.py
```

App opens at `http://localhost:8501` (or similar)

---

## Strava API Setup

1. Create your own Strava API at `https://www.strava.com/settings/api`.
2. Set **Authorization Callback Domain** to `localhost` (the **Website** field is informational; `http://localhost:8501` is fine.)
3. Note your **Client ID** and **Client Secret**.

You don’t need to add a Redirect URI per se; the app uses `http://localhost:8501` and Strava only checks the domain (`localhost`) against your Authorization Callback Domain.

![setup](https://github.com/danielissing/race-predictor/blob/main/images/api_setup.png)
---

## Connecting the App to Strava

In the app’s left sidebar:

1. Enter **Client ID** and **Client Secret** (see image below for where to find them).
2. Click **Connect Strava** and approve.
3. On success, you’ll see **Connected ✅**.

![api_details](https://github.com/danielissing/race-predictor/blob/main/images/strava_api_details.png)

### Save creds (so you don’t retype next time)

* Click **Save app creds** in the sidebar.
* The app stores them in `data/strava_app.json` (plain JSON on your machine).
* Tokens from Strava are stored in `data/strava_tokens.json` and refreshed automatically.

![sidebar1](https://github.com/danielissing/race-predictor/blob/main/images/sidebar_1.png)

> Privacy: All data (creds, tokens, cached activity streams) stays on your computer in the `data/` folder. Nothing is uploaded.

---

## Using the App

### A) 📁 Upload course & set aid stations (🏁 *Upcoming race* tab)

* **GPX**: upload the race route GPX.
* **Aid stations**: paste **cumulative** distances, e.g. `10, 21, 33, 50`.

  * Toggle **km / mi** for the input. All outputs are metric.
* **Conditions**: Simple knob to adjusy for terrain and race-day conditions.

![sidebar2](https://github.com/danielissing/race-predictor/blob/main/images/sidebar_2.png)
#### Course map

* Shows your route over **OpenStreetMap**.
* Aid stations are **merged** if they are very close to one another (default 200 m). If a station is reused, the marker label shows **AS1/AS4**, etc.
* Example: ![course](https://github.com/danielissing/race-predictor/blob/main/images/course_view.png)

#### Segments overview

* Each leg (Start→AS1, AS1→AS2, …) has:

  * an **elevation mini-plot** (click to expand)
  * **length (km)**, **elevation gain/loss (m)**, **min/max elevation**
* Elevation gain/loss uses distance-based resampling and a small hysteresis threshold to avoid over-counting tiny wiggles.
* Example: ![segments](https://github.com/danielissing/race-predictor/blob/main/images/segments.png)

---

### B) 📚 Build your pace curves (📚 *My data* tab)

* **Build from my Strava races**: pulls **Run Type = Race** activities and their **streams**.
* The model bins by grade and computes your **median speed per bin** (with variability from 10th–90th percentile spread).
* The model applies an altitude correction to derive your **sea-level** normalized speed.
* A list of **races used** (name/date/distance) is shown to sanity-check.
* Pace curves and used races are saved to `data/pace_curves.csv` and `data/used_races.csv` so they reload automatically on restart.
* Output: ![pace_curve](https://github.com/danielissing/race-predictor/blob/main/images/pace_curve.png)

### C) Predict ETAs

* Click **Run prediction**.
* You’ll get a table of your ETA, as well as 10th/90th percentile at each station, which you can download.
* Results are cached, so clicking **Download** doesn’t recompute. Recompute only when you click **Run** again or change inputs.
* Output: ![etas](https://github.com/danielissing/race-predictor/blob/main/images/etas.png)

> Notes:
>
> * Streams are **cached** to `data/strava_cache/streams_<activity_id>.json` so you don’t re-hit the API.
> * You can decide to only include races from the last X months and cap a **max number of races** (configurable in code).
 
When clicking **Run prediction**, the code will also be logging intermediate results (times before and after applying various adjustments) to help you understand what's going on if predictions seem wildly off. The output looks like this:

![debug](https://github.com/danielissing/race-predictor/blob/main/images/debug.png)

---

## How It Works (brief)

1. **Pace curves**: For each grade bin (e.g., -12% to -8%), the app looks at your **race streams** and estimates a **median running speed** and a variability factor. It then corrects these median speeds in order to normalize the values to sea level (since you're typically slower when running in high altitude).
2. **Course breakdown**: The GPX is segmented by your aid stations; within each leg, we compute how many meters fall into each grade bin. We also add a penalty (if needed) for each bin to account for the altitude of this particular race.
3. **Fatigue & conditions**: A fatigue curve makes later legs a bit slower and ensures that overall pace drops for races much longer than your median distance. For very long efforts, it also builds in appropriate breaks for aid station time and rest/sleep (currently modeled post hoc to fit the data). A "race day conditions" knob allows to slightly adjust ETAs globally.
4. **Monte Carlo simulation**: We sample speeds around the medians (correlated across bins for “good/bad day” effects) to produce **P10/P90** ETAs.

The [config.py](https://github.com/danielissing/race-predictor/blob/main/config.py) file allows you to fine-tune many parameters to improve predictions. The most important ones are:

* `MAX_RIEGEL_K`: Upper bound for how much slower you'll run in later stages of a race.
* `ULTRA_START_HOURS`: After how many hours the code will start adding rest/sleep time.
* `FATIGUE_SLOPE` and `REST_SLOPE`: Determine how quickly additional time will be added as the race progresses.

---

## Configuration & Persistence

* **Creds**: `data/strava_app.json` (Client ID/Secret)
* **Tokens**: `data/strava_tokens.json` (auto-refreshed)
* **Streams cache**: `data/strava_cache/`
* **Pace curves**: `data/pace_curves.csv`
* **Used races**: `data/used_races.csv`
* **Last ETAs shown**: kept in Streamlit `session_state` (so downloads don’t recompute)

> If you change Strava Client Secret or revoke the app, delete `data/strava_tokens.json` and reconnect.

---

## Troubleshooting

* **Empty/invalid GPX error**
  Make sure you upload a valid GPX. The app reads the file **once** and reuses the bytes everywhere; avoid re-uploading mid-run.

* **High elevation gain**
  The app uses distance resampling + hysteresis (e.g., 20 m step, 3 m threshold). You can tune those in `utils/gpx.py` (`segment_stats`).

* **Strava rate limit (HTTP 429)**
  Wait \~15 minutes. The app caches streams, so already-fetched races won’t re-hit the API.

* **Streams 404 on an activity**
  Some entries (manual, treadmill, privacy settings) don’t have streams. The app **skips** them automatically.

* **Token refresh error (HTTP 400)**
  Your refresh token is invalid (changed app secret or revoked access). Delete `data/strava_tokens.json` and **Connect Strava** again.

* **MediaFileStorageError: Missing PNG**
  Harmless Streamlit quirk when figures are GC’d. We close figures explicitly after rendering; consider upgrading Streamlit if it persists.

---

## License

MIT 

---

## Acknowledgments

Thanks to the Strava API, Streamlit, and the open-source Python community. Happy trails! 🏃‍♂️⛰️

