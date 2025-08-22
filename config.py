"""
Configuration file for Race Time Predictor
All tunable parameters in one place with clear documentation
"""

# ========================================
# FILE PATHS & DIRECTORIES
# ========================================
DATA_DIR = "data"
APP_CREDS_PATH = f"{DATA_DIR}/strava_app.json"  # Strava app credentials
PACE_CURVES_PATH = f"{DATA_DIR}/pace_curves.csv"  # Saved pace model curves
USED_RACES_PATH = f"{DATA_DIR}/used_races.csv"  # Historical races used in model
MODEL_META_PATH = f"{DATA_DIR}/model_meta.json"  # Model metadata
TOKENS_PATH = f"{DATA_DIR}/strava_tokens.json"  # OAuth tokens
CACHE_DIR = f"{DATA_DIR}/strava_cache"  # API response cache

# ========================================
# STRAVA API CONFIGURATION
# ========================================
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"
DEFAULT_TIMEOUT = 30  # API request timeout in seconds
MAX_ACTIVITIES = 200  # Maximum number of races to fetch from Strava

# ========================================
# PACE MODEL PARAMETERS
# ========================================

# Grade bins for pace curves (% grade)
# More bins = more accurate but needs more data
GRADE_BINS = [-50, -30, -20, -12, -8, -5, -3, 0, 3, 5, 8, 12, 20, 30, 50]

# Default Riegel exponent (how pace degrades with distance)
# 1.06 is standard for road races
DEFAULT_RIEGEL_K = 1.06
MIN_RIEGEL_K = 0.8
MAX_RIEGEL_K = 1.3
ROAD_K_DAMPEN = 0.8 # moves calculates Riegel_k for road races closer to 1

# Recency weighting for historical races
# How many months before a race's weight is halved
MILD_HALF_LIFE = 18.0  # months (1.5 years)

# ========================================
# ALTITUDE ADJUSTMENT PARAMETERS
# ========================================

# Performance reduction at altitude (fraction per 1000m above threshold)
# 0.06 = 6% slower per 1000m elevation
# Based on sports science literature
ELEVATION_IMPAIRMENT = 0.06

# Elevation where performance starts to degrade (meters)
# Sea level to 300m has negligible effect
ELEVATION_IMPAIRMENT_THRESHOLD = 300.0

# ========================================
# MONTE CARLO SIMULATION PARAMETERS
# ========================================

# Number of simulations for confidence intervals
# More = more accurate percentiles but slower
MC_SIMS = 500

# Day-to-day performance variability (coefficient of variation)
# Accounts for sleep, nutrition, minor illness, etc.
SIGMA_DAY = 0.06

# Correlation between grade bins in performance variation
# 0.4 = if you're slow uphill, you're somewhat likely slow downhill too
# 0 = independent, 1 = perfectly correlated
RHO = 0.4

# Relative uncertainty bounds for pace at each grade
# How much your pace varies at a specific grade
SIGMA_REL_LOW = 0.07  # Minimum uncertainty
SIGMA_REL_HIGH = 0.5  # Maximum uncertainty
SIGMA_REL_DEFAULT = 0.2  # Default when no data

# ========================================
# ULTRA RACE ADJUSTMENTS
# ========================================

# When to start applying ultra-specific adjustments (hours)
ULTRA_START_HOURS = 5.0

# Special adjustments for extreme races
SHORT_DISTANCE_KM = 30 # threshold for short races
DEFAULT_DISTANCE_KM = 50.0 # median race
MEDIUM_DISTANCE_KM = 80 #threshold for medium races
EXTREME_DISTANCE_KM = 200  # Apply extra penalties above this
MOUNTAIN_DISTANCE_KM = 250  # Mountain ultra threshold

# for ultra adjustments
EXTREME_DISTANCE_FACTOR = 1.3  # Extra rest time multiplier
EXTREME_FATIGUE_FACTOR = 1.08  # Extra fatigue multiplier
SHORT_REST_PCT = 0.01
MED_REST_PCT = 0.05
LONG_REST_PCT = 0.15

# Modeling fatigue and rest
FATIGUE_SLOPE = 0.0005 # Linear increase of fatigue factor based on race time
REST_SLOPE = 0.1  # Linear increase in rest time based on race time

#Modeling variance
DEFAULT_VARIANCE = 0.12 # relative variance for short races
MAX_VARIANCE = 0.15 # cap for relative variance when predicting arrival times
MIN_VARIANCE = 0.09 # see max var
EXTREME_VARIANCE = 0.35
VARIANCE_EXP = 0.05 # multiplicative factor for exponential term in variance function
REBOUND = 0.02 # multiplicative factor for rebound term in variance function

# --- Day model (mixture) ---
PROB_NORMAL   = 0.70
PROB_OFF      = 0.25
PROB_OUTLIER  = 0.05
OUTLIER_GOOD_SHARE = 0.30   # fraction of outliers that are "amazing"

# Noise scales are multiples of relative variance
NORMAL_SIGMA_MULT   = 0.40  # N(0, 0.4*RV)
OFF_SIGMA_MULT      = 0.60  # half-normal(+)
AMAZING_SIGMA_MULT  = 0.80  # half-normal(-)
LOGNORM_SIGMA       = 0.60  # lognormal tail spread

# Condition effects
COND_MEAN_PER_UNIT = 0.02   # ±2% per unit per step on the "race codnitions" slider
COND_VAR_PER_UNIT  = 0.10   # narrows/widens spread by ±10%/unit

# Grade variation
GRADE_VAR_ROAD  = 0.30
GRADE_VAR_TRAIL = 0.40

# Tail cap derived from RV (keeps disasters realistic without a separate function)
TAILCAP_BASE_ROAD   = 0.10
TAILCAP_BASE_TRAIL  = 0.12
TAILCAP_K_ROAD      = 2.0
TAILCAP_K_TRAIL     = 2.5
TAILCAP_MIN         = DEFAULT_VARIANCE
TAILCAP_MAX         = EXTREME_VARIANCE

# Progress exponents
PROGRESS_EXP_SHORT = 0.3 
PROGRESS_EXP_MEDIUM = 0.5
PROGRESS_EXP_LONG = 0.7
PROGRESS_EXP_ULTRA = 0.9

# ========================================
# GPX PROCESSING PARAMETERS
# ========================================

# Earth radius for distance calculations (meters)
EARTH_R = 6371000.0

# Step size for resampling GPX tracks (meters)
# Smaller = more accurate but larger files
STEP_LENGTH = 10.0

# Window size for smoothing elevation data (meters)
# Larger = smoother but may miss short steep sections
STEP_WINDOW = 40.0
DEFAULT_SMOOTH_WINDOW = 5  # Points for additional smoothing

# Distance for clustering aid stations (meters)
# Aid stations within this distance are grouped together
CLUSTER_RADIUS = 200.0

# Road race detection threshold
ROAD_GAIN_PER_KM = 10  # Less than this = road race (m/km)

# ========================================
# PHYSICAL LIMITS & SAFETY BOUNDS
# ========================================

# Minimum speed to consider "moving" (m/s)
# Below this is considered stopped
SEA_LEVEL_CLIP_LOW = 0.05  # ~72 min/km pace

# Maximum reasonable running speed (m/s)
# Above this is likely GPS error
SEA_LEVEL_CLIP_HIGH = 6.0  # ~2:47 min/km pace

# Small number to prevent division by zero
EPSILON = 1e-6

# ========================================
# UNIT CONVERSIONS
# ========================================
MILES_TO_KM = 1.609344
SECONDS_PER_HOUR = 3600
MINUTES_PER_HOUR = 60
MONTH_LENGTH = 30.437  # Average days per month
