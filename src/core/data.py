import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def preprocess_production(df, date_col='DATEPRD', prod_col='BORE_OIL_VOL',
                          smooth_window=14):
    """
    Convert a raw field CSV into a clean daily-rate DataFrame for LSTM.

    Strict pipeline order (must NOT be reordered):
    ------------------------------------------------
    Step 1 : Parse & sort chronologically
    Step 2 : Auto-detect cumulative vs daily rate
             If >50% of consecutive non-zero diffs are positive → cumulative.
             Apply .diff() to convert cumulative volume → daily rate.
    Step 3 : dropna()   — removes NaN introduced by diff() at row 0
    Step 4 : Drop zeros and negatives  — shut-in / noise rows
    Step 5 : Rolling-mean smoother (causal, min_periods=1, window=14)
             ALWAYS applied AFTER diff, NEVER before.

    Returns
    -------
    cleaned_df     : pd.DataFrame  columns ['DATEPRD', 'BORE_OIL_VOL']  (rate/day)
    was_cumulative : bool          True if .diff() was applied
    """
    # ── Step 1 : Parse & sort ───────────────────────────────────────────────
    work = df[[date_col, prod_col]].copy()
    work.columns = ['DATEPRD', 'BORE_OIL_VOL']
    work['DATEPRD']      = pd.to_datetime(work['DATEPRD'])
    work['BORE_OIL_VOL'] = pd.to_numeric(work['BORE_OIL_VOL'], errors='coerce')
    work = work.dropna().sort_values('DATEPRD').reset_index(drop=True)

    # ── Step 2 : Detect cumulative vs daily rate ────────────────────────────
    # Use only the non-zero values to judge monotonicity.
    vals          = work['BORE_OIL_VOL'].values
    nz_vals       = vals[vals > 0]
    was_cumulative = False

    if len(nz_vals) > 10:
        diffs      = np.diff(nz_vals.astype(np.float64))
        pct_rising = float(np.mean(diffs > 0))
        # >50% rising → monotonically increasing → cumulative volume
        if pct_rising > 0.50:
            work['BORE_OIL_VOL'] = work['BORE_OIL_VOL'].diff()  # cumulative → daily rate
            was_cumulative = True

    # ── Step 3 : Drop NaN introduced by diff() ─────────────────────────────
    work = work.dropna()

    # ── Step 4 : Drop zeros and negatives ──────────────────────────────────
    work = work[work['BORE_OIL_VOL'] > 0].reset_index(drop=True)

    # ── Step 5 : Smooth (causal, no look-ahead) — ALWAYS after diff ────────
    if len(work) >= smooth_window:
        work['BORE_OIL_VOL'] = (
            work['BORE_OIL_VOL']
            .rolling(window=smooth_window, min_periods=1)
            .mean()
        )

    return work.reset_index(drop=True), was_cumulative


def calculate_metrics(actual, predicted):
    """
    Compute RMSE, MAE, MAPE, R², NSE on original-scale (inverse-transformed) values.
    R² is computed via sklearn.metrics.r2_score — NOT on scaled values.
    """
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask      = np.isfinite(actual) & np.isfinite(predicted)
    act, pred = actual[mask], predicted[mask]

    if len(act) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan, 'NSE': np.nan}

    rmse = float(np.sqrt(mean_squared_error(act, pred)))
    mae  = float(mean_absolute_error(act, pred))

    nz   = act != 0
    mape = float(np.mean(np.abs((act[nz] - pred[nz]) / act[nz])) * 100) \
           if nz.any() else 0.0

    # R² — sklearn r2_score on original (inverse-transformed) scale values
    r2  = float(r2_score(act, pred))

    # Nash-Sutcliffe Efficiency (identical formula to R² for this use-case)
    ss_res = np.sum((act - pred) ** 2)
    ss_tot = np.sum((act - act.mean()) ** 2)
    nse    = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'NSE': nse}


def generate_sample_data():
    """Synthetic daily-rate Arps-decline data (NOT cumulative)."""
    dates = pd.date_range(start='2020-01-01', periods=800, freq='D')
    t     = np.arange(len(dates))

    base        = 8_000 * np.exp(-0.0015 * t)
    noise       = np.random.normal(0, 150, len(dates))
    seasonality = 300  * np.sin(2 * np.pi * t / 365.25)
    production  = np.clip(base + noise + seasonality, 100, None)

    shut_in = np.random.choice(t, size=8, replace=False)
    production[shut_in] = 0

    return pd.DataFrame({
        'DATEPRD':               dates,
        'BORE_OIL_VOL':          production,
        'AVG_DOWNHOLE_PRESSURE': 300 * np.exp(-0.0005 * t) + np.random.normal(0, 10, len(dates)),
        'AVG_WHP_P':             100 * np.exp(-0.001  * t) + np.random.normal(0,  5, len(dates)),
        'Total Water Injection':  2_000 + 5 * t             + np.random.normal(0, 100, len(dates)),
    })


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test: compares forecast errors of two models.

    Parameters
    ----------
    e1 : array-like  errors from model 1 (e.g. Hybrid LSTM)
    e2 : array-like  errors from baseline model
    h  : int         forecast horizon (default 1)

    Returns
    -------
    dict with keys: dm_stat, p_value, significant, interpretation
    """
    from scipy import stats as _stats

    e1 = np.array(e1, dtype=float)
    e2 = np.array(e2, dtype=float)
    d  = e1 ** 2 - e2 ** 2          # loss differential (squared errors)
    n  = len(d)

    if n < 3:
        return {'dm_stat': np.nan, 'p_value': np.nan,
                'significant': False, 'interpretation': 'Insufficient data'}

    d_mean = np.mean(d)
    # Newey-West HAC variance estimate (lag = h - 1)
    gamma0 = np.var(d, ddof=1)
    hac_var = gamma0
    for lag in range(1, h):
        gamma_l = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        hac_var += 2 * (1 - lag / h) * gamma_l
    hac_var = max(hac_var, 1e-12)

    dm_stat  = d_mean / np.sqrt(hac_var / n)
    p_value  = float(2 * (1 - _stats.norm.cdf(abs(dm_stat))))
    sig      = p_value < 0.05
    interp   = ("Hybrid LSTM significantly better" if sig and dm_stat > 0
                else "Baseline significantly better" if sig and dm_stat < 0
                else "No significant difference")

    return {'dm_stat': float(dm_stat), 'p_value': p_value,
            'significant': sig, 'interpretation': interp}
