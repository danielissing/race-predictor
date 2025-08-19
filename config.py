# --- App Configuration ---
DATA_DIR = "data"
APP_CREDS_PATH = f"{DATA_DIR}/strava_app.json"
PACE_CURVES_PATH = f"{DATA_DIR}/pace_curves.csv"
USED_RACES_PATH = f"{DATA_DIR}/used_races.csv"
MODEL_META_PATH = f"{DATA_DIR}/model_meta.json"

# --- Strava API Configuration ---
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"
TOKENS_PATH = f"{DATA_DIR}/strava_tokens.json"
CACHE_DIR = f"{DATA_DIR}/strava_cache"
DEFAULT_TIMEOUT = 30

# --- Model & Course Parameters ---
GRADE_BINS = [-100, -30, -20, -12, -8, -5, -3, 0, 3, 5, 8, 12, 20, 30, 100]
MC_SIMS = 1000
DEFAULT_RIEGEL_K = 1.06
STEP_LENGTH = 10.0
STEP_WINDOW = 40.0
MAX_ACTIVITIES = 200
CLUSTER_RADIUS = 200.0
EPSILON = 1e-6
HEAT_MULT={'cool':1.00,'moderate':1.02,'hot':1.05}
FEEL_MULT={'good':0.98,'ok':1.00,'meh':1.02}
SIGMA_DAY = 0.035
RHO = 0.4
SLEEP_CUTOFF = 12.0

# --- GPX & Ultra Adjustment Parameters ---
EARTH_R = 6371000.0
ELEVATION_IMPAIRMENT = 0.06 # % penalty for high elevation races
ELEVATION_IMPAIRMENT_THRESHOLD = 300.0 # elevation from which on elevation penalty is applied
MONTH_LENGTH = 30.437
REST_PER_24H = 2.0
ULTRA_GAMMA = 0.30
START_TH_H = 20.0
MILD_HALF_LIFE = 18.0
MILES_TO_KM = 1.609344
SECONDS_PER_HOUR = 3600
MINUTES_PER_HOUR = 60
SEA_LEVEL_CLIP_LOW = 0.1
SEA_LEVEL_CLIP_HIGH = 6.0
SIGMA_REL_LOW = 0.05
SIGMA_REL_HIGH = 1.0
SIGMA_REL_DEFAULT = 0.1
DEFAULT_SMOOTH_WINDOW = 5