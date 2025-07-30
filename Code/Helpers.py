from __future__ import annotations

import numpy as np
import pandas as pd
import math
from scipy.interpolate import UnivariateSpline
from typing import Tuple

def estimate_v0(prev_interval_ballspeed: list[float], ms_since_release: list[float],
                drag_coefficient=0.001597) -> list[float]:
    """
    Reverse-engineer initial velocity from the average velocity recorded
    over an interval_length in flight.
    Assumes prev_interval_ballspeed is in ft / s.
    Returns NaN where the quadratic has no real solution.

    Credits to Scott Powers for the drag coeffient
    https://github.com/saberpowers/predictive-pitch-score/blob/main/scripts/sandbox/2024-06-04_infer_trajectory.R
    """
    # --- 1.  Convert to plain NumPy & correct units -------------------------
    v_avg = np.asarray(prev_interval_ballspeed, dtype=float)
    t_sec = np.asarray(ms_since_release, dtype=float) * 1e-3   # ms → s

    # --- 2.  Quadratic inversion -------------------------------------------
    A     = 0.5 * drag_coefficient * t_sec
    disc  = 1.0 - 4.0 * A * v_avg

    # Only keep the physical (smaller) root where disc ≥ 0
    valid = disc > 0
    v0    = np.full_like(v_avg, np.nan)        # initialise with NaN
    v0[valid] = (1.0 - np.sqrt(disc[valid])) / (2.0 * A[valid])

    return v0

def batter_outcome(batter: list[str], 
                   next_first_baserunner: list[str], next_second_baserunner: list[str], next_third_baserunner: list[str]
                   ) -> list[str]:
    batter = pd.Series(batter)
    nfb = pd.Series(next_first_baserunner)
    nsb = pd.Series(next_second_baserunner)
    ntb = pd.Series(next_third_baserunner)

    result = pd.Series(['out']*len(batter)) # Default is out
    safe_mask = (nfb == batter)
    na_mask = (nsb == batter) | (ntb == batter)

    result[safe_mask] = "safe"
    result[~safe_mask & na_mask] = "NA"

    return result.tolist()

def to_list(x):
    """Coerce every value to a list. NaN/None → empty list."""
    # 1 – already a list
    if isinstance(x, list):
        return x

    # 2 – iterable types we want to flatten
    if isinstance(x, (tuple, set, np.ndarray)):
        return list(x)

    # 3 – missing values (None, NaN, pandas NA)
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []

    # 4 – every other scalar → wrap as singleton list
    return [x]

# Helper Function for Infield
# Infield circle is defined as a 95ft arc centered
# at the pitching rubber
def is_in_infield(x, y, scale=95):
    if pd.isna(x) or pd.isna(y):
        return False
    dist = np.sqrt((x-0)**2 + (y-60.5)**2)
    angle_deg = np.degrees(np.arctan2(y,x)) # Angle above x-axis
    return (dist <= scale) and (45 <= angle_deg <= 135) # Bool

# Helper function for Running.ipynb
def make_smoothing_spline(runpos: pd.DataFrame, k: int = 3
                          ) -> Tuple[UnivariateSpline, pd.DataFrame, float]:
    """
    Fit a cubic (k=3) smoothing spline to 1-D position data.

    Parameters:
        runpos : pd.DataFrame
            Must contain 'timestamp' and 'ft_til_1st'.
            ('timestamp' MUST BE MONOTONIC)
        k : int, default 3
            Polynomial degree (3 = cubic). 2 for quadratic, 5 for quintic ….
    Returns:
        spline : scipy.interpolate.UnivariateSpline
            Fitted spline object; call `spline(t_query)` for arbitrary times.
        smoothed : pd.DataFrame
            Copy of `runpos` with new column 'ft_til_1st_sm'.
        rmse : float
            Indicates "noisyness" of the sample
    """
    df = runpos.copy()

    ## 1. Prepare
    t = df['timestamp']
    df["t"] = t
    y = df["ft_til_1st"].values
    t = df["t"].values
    n = len(t)

    ## 2. Choose smoothing factor
    # Smoothing factor *s* passed to `UnivariateSpline`.
    #  - Smaller s  ⇒ closer fit (less smoothing).
    #  - Larger  s  ⇒ smoother curve (more bias).
    s_factor = n * np.var(y) * 0.5e-4  
    # (≈ 0.005 × RMSE of raw noise if noise σ ≈ 1 ft)

    ## 3. Fit spline
    spline = UnivariateSpline(t, y, k=k, s=s_factor)

    ## 4. Evaluate spline at original sample points 
    df["ft_til_1st_sm"] = spline(t)

    ## 5. Diagnostic metrics
    df["abs_error_raw_vs_smooth"] = np.abs(y - df["ft_til_1st_sm"].values)
    rmse = np.sqrt(np.mean(df["abs_error_raw_vs_smooth"] ** 2))

    print(f"[make_smoothing_spline] fitted cubic spline with s={s_factor:.3g}, "
          f"RMSE(raw→smooth)={rmse:.4f} ft over {n} samples.")

    return spline, df, rmse

# Helper class for Running.ipynb
# (Define Ensemble class for simplicity (polymorphism))
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
class Ensemble:
    def __init__(self, models: list):
        self.models = models
    def predict(self, X, y=None, dtrain=None):
        preds = []
        for model in self.models:
            if isinstance(model, xgb.Booster):
                dmat = xgb.QuantileDMatrix(X, label=y, ref=dtrain)
                preds.append(model.predict(dmat))
            elif isinstance(model, cat.CatBoostRegressor):
                preds.append(model.predict(X))
            elif isinstance(model, lgb.Booster):
                preds.append(model.predict(X))
            else:
                raise TypeError("Invalid class")
        return np.mean(preds, axis=0)
    def predict_proba(self, X, y=None, dtrain=None):
        prob_preds = []
        for model in self.models:
            if isinstance(model, xgb.Booster):
                # "multi:softprob"
                dmat = xgb.DMatrix(X, label=y)
                prob_preds.append(model.predict(dmat))
            elif isinstance(model, cat.CatBoostClassifier):
                prob_preds.append(model.predict(X, prediction_type="Probability"))
            elif isinstance(model, lgb.Booster):
                prob_preds.append(model.predict(X))  
            else:
                raise TypeError("Invalid class")
        return np.mean(prob_preds, axis=0)