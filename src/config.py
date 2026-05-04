"""
PetroAI Suite — Central Hyperparameter Configuration
All model and physics constants are defined here and imported project-wide.
"""

# ── Sequence / Training defaults ──────────────────────────────────────────────
LOOKBACK: int           = 30       # default lookback window (days)
DROPOUT_RATE: float     = 0.2      # MC Dropout rate
LSTM_UNITS_1: int       = 32       # first LSTM layer units
LSTM_UNITS_2: int       = 16       # second LSTM layer units (LSTM_UNITS_1 // 2)
MC_ITERATIONS: int      = 100      # Monte Carlo Dropout inference passes

# ── Physics-informed loss weights ─────────────────────────────────────────────
ALPHA_PHYSICS: float    = 0.3      # weight on Arps penalty term L_physics
BETA_MONOTONE: float    = 0.1      # weight on monotone decline penalty L_monotone

# ── Adaptive blending ─────────────────────────────────────────────────────────
LAMBDA_ADAPTIVE: float  = 1.0      # λ — adaptive weight sensitivity
BLEND_WINDOW: int       = 30       # taper window for anchor offset

# ── Post-processing constants ─────────────────────────────────────────────────
SMOOTHING_SPAN: int     = 7        # EWM span for final smoothing
RISE_CAP: float         = 0.005    # max per-step production increase (0.5 %)
FLOOR_FRACTION: float   = 0.005    # floor = last_actual × FLOOR_FRACTION
CEILING_FRACTION: float = 1.50     # ceiling = last_actual × CEILING_FRACTION

# ── Petroleum engineering ─────────────────────────────────────────────────────
ECONOMIC_LIMIT_BBL: float = 10.0   # abandonment rate (bbl/day) for EUR calc
N_RECENT_DECLINE: int     = 90     # recent rows used to fit Arps decline

# ── Uncertainty calibration thresholds ────────────────────────────────────────
PICP_TARGET: float      = 0.80     # ideal 80-% prediction interval coverage
PICP_LOW: float         = 0.70     # below → overconfident
PICP_HIGH: float        = 0.90     # above → too conservative
