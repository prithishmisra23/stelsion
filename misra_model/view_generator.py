# misra_model/view_generator.py

import numpy as np
from .config import CONFIG


def bin_flux(phase, flux, bin_edges):
    """Bin flux values into phase bins using median"""
    n_bins = len(bin_edges) - 1
    binned = np.ones(n_bins)  # Default to 1.0 (baseline)
    
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if mask.sum() > 0:
            binned[i] = np.median(flux[mask])
    
    return binned


def generate_global_view(phase, flux, n_bins=2001):
    """
    2001-point global view covering full orbital period.
    AstroNet standard — gives context about full orbit.
    Shape: (2001,) → fed to InceptionTime 1D branch
    """
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    return bin_flux(phase, flux, bin_edges).astype(np.float32)


def generate_local_view(phase, flux, duration, period, n_bins=201, width_factor=4):
    """
    201-point local view zoomed in to transit region.
    Shows 4 transit durations on each side — captures ingress/egress shape.
    Shape: (201,) → fed to simple 1D CNN branch
    """
    half_width = width_factor * duration / period
    bin_edges = np.linspace(-half_width, half_width, n_bins + 1)
    
    # Only use points within the window
    mask = (phase >= -half_width) & (phase <= half_width)
    phase_local = phase[mask]
    flux_local = flux[mask]
    
    if len(phase_local) < 5:
        return np.ones(n_bins, dtype=np.float32)
    
    return bin_flux(phase_local, flux_local, bin_edges).astype(np.float32)


def generate_2d_matrix(time, flux, period, t0, n_orbits=10, n_phase_bins=200):
    """
    YOUR KEY CONTRIBUTION — 2D folded matrix.
    
    Shape: (10, 200) — 10 rows (individual orbits) × 200 columns (phase bins)
    
    Why this matters:
    - Each ROW is one complete orbit's worth of data
    - Transit signal appears as a VERTICAL DARK BAND when period is exact
    - When period is slightly off, band TILTS — 2D-CNN still detects it
    - This tolerates period uncertainty up to 20% (proven in 1904.12419 paper)
    - A 1D-CNN on averaged phase-fold LOSES this information
    
    This is the differentiating feature of your model.
    """
    phase_bin_edges = np.linspace(-0.5, 0.5, n_phase_bins + 1)
    matrix = np.ones((n_orbits, n_phase_bins), dtype=np.float32)
    
    for orbit_idx in range(n_orbits):
        # Time window for this orbit
        t_start = t0 + orbit_idx * period
        t_end = t0 + (orbit_idx + 1) * period
        
        orbit_mask = (time >= t_start) & (time < t_end)
        
        if orbit_mask.sum() < 3:
            continue
        
        orbit_time = time[orbit_mask]
        orbit_flux = flux[orbit_mask]
        
        # Convert to phase for this orbit
        orbit_phase = ((orbit_time - t_start) / period) - 0.5
        
        # Bin into phase columns
        matrix[orbit_idx] = bin_flux(orbit_phase, orbit_flux, phase_bin_edges)
    
    return matrix


def generate_all_views(time, flux, period_info, config=CONFIG):
    """
    Generate all three views from a single light curve.
    
    Input:
        time: array of observation times (days)
        flux: preprocessed normalized flux array
        period_info: dict from period_finder.find_best_period()
    
    Returns:
        global_view: (2001,) float32 array
        local_view: (201,) float32 array  
        matrix_2d: (10, 200) float32 array
    """
    from .period_finder import phase_fold
    
    period = period_info['period']
    t0 = period_info['t0']
    duration = period_info['duration']
    
    # Phase fold
    phase, flux_folded = phase_fold(time, flux, period, t0)
    
    # Generate views
    global_view = generate_global_view(
        phase, flux_folded,
        n_bins=config["global_bins"]
    )
    local_view = generate_local_view(
        phase, flux_folded, duration, period,
        n_bins=config["local_bins"],
        width_factor=config["local_transit_widths"]
    )
    matrix_2d = generate_2d_matrix(
        time, flux, period, t0,
        n_orbits=config["matrix_orbits"],
        n_phase_bins=config["matrix_phase_bins"]
    )
    
    return global_view, local_view, matrix_2d
