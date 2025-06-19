import numpy as np
import pandas as pd

def estimate_v0(prev_interval_ballspeed: list[float], ms_since_release: list[float],
                drag_coefficient=0.001597) -> list[float]:
    """
    Reverse-engineer initial velocity from the average velocity recorded
    over an interval_length in flight.
    Assumes prev_interval_ballspeed is in ft / s.
    Returns NaN where the quadratic has no real solution.

    Credits to Scott Powers for the drag coeffient
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
