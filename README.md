# Trail Running ETA Predictor

A local Streamlit app that predicts **arrival times plus 10th/90th percentile** at each aid station for a trail race.
It uses your **Strava race history** to learn a personal **speed vs. grade** curve, combines it with a **GPX** of the course (including elevation), and includes a simple simulation to account for fatigue and conditions.

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

Python packages (installed via `requirements.txt`):

* `streamlit`, `pandas`, `numpy`, `requests`
* `gpxpy` (GPX parsing), `matplotlib` (elevation plots)
* `folium`, `streamlit-folium` (course map)

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

> Privacy: All data (creds, tokens, cached activity streams) stays on your computer in the `data/` folder. Nothing is uploaded.

---

## Using the App

### A) 📁 Upload course & set aid stations (🏁 *Upcoming race* tab)

* **GPX**: upload the race route GPX.
* **Aid stations**: paste **cumulative** distances, e.g. `10, 21, 33, 50`.

  * Toggle **km / mi** for the input. All outputs are metric.
* **Conditions**: choose **Heat** (cool/moderate/hot) and **Feeling** (good/ok/meh) to adjust for race-day conditions.

#### Course map

* Shows your route over **OpenStreetMap**.
* Aid stations are **merged** if they are very close to one another (default 200 m). If a station is reused, the marker label shows **AS1/AS4**, etc.

#### Segments overview

* Each leg (Start→AS1, AS1→AS2, …) has:

  * an **elevation mini-plot**
  * **length (km)**, **elevation gain/loss (m)**, **min/max elevation**
* Elevation gain/loss uses distance-based resampling and a small hysteresis threshold to avoid over-counting tiny wiggles (closer to watch values).

---

### B) 📚 Build your pace curves (📚 *My data* tab)

* **Build from my Strava races**: pulls **Run Type = Race** activities and their **streams**.
* The model bins by grade and computes your **median speed per bin** (with variability from 10th–90th percentile spread).
* A list of **races used** (name/date/distance) is shown to sanity-check.
* Pace curves and used races are saved to `data/pace_curves.csv` and `data/used_races.csv` so they reload automatically on restart.

#### C) Predict ETAs

* Click **Run prediction**.
* You’ll get a table of **ETA P10 / P50 / P90** at each station and can **download CSV**.
* Results are cached, so clicking **Download** doesn’t recompute. Recompute only when you click **Run** again or change inputs.

> Notes:
>
> * Streams are **cached** to `data/strava_cache/streams_<activity_id>.json` so you don’t re-hit the API.
> * You can limit to the **last 24 months** and cap a **max number of races** (configurable in code).

---

## How It Works (brief)

1. **Pace curves**: For each grade bin (e.g., -12% to -8%), the app looks at your **race streams** and estimates a **median running speed** and a variability factor.
2. **Course breakdown**: The GPX is segmented by your aid stations; within each leg, we compute how many meters fall into each grade bin.
3. **Fatigue & conditions**: A simple fatigue curve makes later legs a bit slower; heat/feeling multipliers adjust globally.
4. **Monte Carlo**: We sample speeds around the medians (correlated across bins for “good/bad day” effects) to produce **P10/P50/P90** ETAs.

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

