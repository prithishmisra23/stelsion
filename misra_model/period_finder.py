# misra_model/period_finder.py

import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u


def find_best_period(time, flux, config=None):
    """
    BLS (Box Least Squares) period search.
    Runs BEFORE the neural network.
    This is what finds the transit period on blind/unlabeled data.
    
    Returns:
        period (float): best period in days
        t0 (float): time of first transit center
        duration (float): transit duration in days
        depth (float): transit depth (fractional)
        power (float): BLS power at best period (SNR proxy)
    """
    if config is None:
        from .config import CONFIG
        config = CONFIG
    
    # Build period grid
    periods = np.linspace(
        config["period_min"],
        config["period_max"],
        config["n_periods"]
    )
    
    # Run BLS
    bls = BoxLeastSquares(time * u.day, flux)
    
    # Duration grid: from 1 hour to 10.8 hours
    durations = np.array([0.04, 0.08, 0.1, 0.15, 0.2, 0.3, 0.45]) * u.day
    
    result = bls.power(periods * u.day, durations)
    
    # Find best peak
    best_idx = np.argmax(result.power)
    
    best_period = float(result.period[best_idx].value)
    best_t0 = float(result.transit_time[best_idx].value)
    best_duration = float(result.duration[best_idx].value)
    best_depth = float(result.depth[best_idx])
    best_power = float(result.power[best_idx])
    
    return {
        'period': best_period,
        't0': best_t0,
        'duration': best_duration,
        'depth': best_depth,
        'bls_power': best_power
    }


def phase_fold(time, flux, period, t0):
    """
    Phase fold a light curve at a given period.
    Returns phase array from -0.5 to 0.5 with transit at phase 0.
    """
    phase = ((time - t0) % period) / period
    
    # Center transit at phase 0 (move from [0,1] to [-0.5, 0.5])
    phase[phase > 0.5] -= 1.0
    
    # Sort by phase for clean visualization
    sort_idx = np.argsort(phase)
    
    return phase[sort_idx], flux[sort_idx]
