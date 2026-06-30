# misra_model/preprocessor.py

import numpy as np
from scipy.signal import savgol_filter
import pywt


def interpolate_nans(flux):
    """Fill NaN gaps by linear interpolation"""
    nan_mask = np.isnan(flux)
    if nan_mask.any():
        indices = np.arange(len(flux))
        flux[nan_mask] = np.interp(
            indices[nan_mask],
            indices[~nan_mask],
            flux[~nan_mask]
        )
    return flux


def sigma_clip(flux, sigma=3.0, n_iter=5):
    """Iteratively remove outliers beyond sigma standard deviations"""
    mask = np.ones(len(flux), dtype=bool)
    for _ in range(n_iter):
        median = np.median(flux[mask])
        std = np.std(flux[mask])
        mask = np.abs(flux - median) < sigma * std
    
    # Replace outliers with local median instead of removing
    flux_clean = flux.copy()
    if not mask.all():
        indices = np.arange(len(flux))
        flux_clean[~mask] = np.interp(
            indices[~mask],
            indices[mask],
            flux[mask]
        )
    return flux_clean


def savgol_detrend(flux, window=301, polyorder=3):
    """Savitzky-Golay smoothing for low-noise light curves"""
    if window > len(flux):
        window = len(flux) // 2
        if window % 2 == 0:
            window -= 1
    trend = savgol_filter(flux, window_length=window, polyorder=polyorder)
    return flux / trend


def wavelet_detrend(flux, wavelet='db4', level=4):
    """
    Wavelet denoising for high-noise light curves.
    Preserves sharp transit edges while removing high-frequency noise.
    Uses db4 (Daubechies 4) wavelet — standard for astronomical signals.
    """
    coeffs = pywt.wavedec(flux, wavelet, level=level)
    
    # Threshold high-frequency coefficients (detail levels)
    # Keep approximation coefficients (trend) for detrending
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(flux)))
    
    # Apply soft thresholding to detail coefficients only
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for detail in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(detail, threshold, mode='soft'))
    
    flux_denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    # Trim to original length (waverec may add one sample)
    flux_denoised = flux_denoised[:len(flux)]
    
    return flux / (flux_denoised + 1e-10)


def normalize(flux):
    """Normalize so baseline median = 1.0"""
    median = np.median(flux)
    if median == 0:
        return flux
    return flux / median


def estimate_noise_level(flux):
    """
    Estimate noise using normalized median absolute deviation (NMAD).
    High noise → use wavelet detrending.
    Low noise → use Savitzky-Golay.
    """
    return 1.4826 * np.median(np.abs(np.diff(flux)))


def preprocess(time, flux, method='adaptive', config=None):
    """
    Full preprocessing pipeline:
    1. NaN imputation
    2. Sigma clipping  
    3. Adaptive detrending (SG or Wavelet based on noise level)
    4. Normalization
    
    Returns: cleaned time array, cleaned flux array
    """
    # Step 1: Fill NaNs
    flux = interpolate_nans(flux.copy())
    
    # Step 2: Remove outliers
    flux = sigma_clip(flux, sigma=3.0)
    
    # Step 3: Detrend
    if method == 'adaptive':
        noise = estimate_noise_level(flux)
        noise_threshold = 0.005  # 0.5% noise level
        if noise > noise_threshold:
            flux = wavelet_detrend(flux)
        else:
            flux = savgol_detrend(flux)
    elif method == 'savgol':
        flux = savgol_detrend(flux)
    elif method == 'wavelet':
        flux = wavelet_detrend(flux)
    
    # Step 4: Normalize
    flux = normalize(flux)
    
    return time, flux
