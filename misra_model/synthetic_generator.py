import os
import glob
import numpy as np
from batman import TransitModel, TransitParams

def inject_transit(time, flux, period, t0, rp_rs, a_rs, inc=89.0):
    """
    Inject a physically realistic transit into a real light curve.
    Uses batman (Bad-Ass Transit Model cAlculatioN) for accurate physics.
    """
    params = TransitParams()
    params.t0      = t0          # time of inferior conjunction
    params.per     = period      # orbital period
    params.rp      = rp_rs       # planet radius / star radius
    params.a       = a_rs        # semi-major axis / star radius
    params.inc     = inc         # orbital inclination (degrees)
    params.ecc     = 0           # eccentricity
    params.w       = 90          # longitude of periastron
    params.limb_dark = "quadratic"
    params.u       = [0.4, 0.26] # limb darkening coefficients
    
    m = TransitModel(params, time)
    transit_flux = m.light_curve(params)
    
    return flux * transit_flux


def generate_synthetics(num_samples=5000, background_dir='../dataset'):
    os.makedirs('dataset_synthetic', exist_ok=True)
    
    # Load real non-planet light curves as backgrounds
    # For this script, we will grab all npz files from the target dataset folder
    non_planet_files = glob.glob(os.path.join(background_dir, '*.npz'))
    
    if len(non_planet_files) == 0:
        print(f"Error: No background .npz files found in {background_dir}. Please ensure Kepler data exists first.")
        return

    synthetic_labels = []
    
    print(f"Generating {num_samples} synthetic planet light curves...")

    for i in range(num_samples):
        # Pick random background light curve
        bg_file = np.random.choice(non_planet_files)
        data = np.load(bg_file)
        time = data['time']
        flux = data['flux']
        
        # Avoid extremely short/corrupt files
        if len(time) < 100:
            continue
            
        # Random physical parameters
        period = np.random.uniform(1.0, 20.0)
        t0     = time[0] + np.random.uniform(0, period)
        rp_rs  = np.random.uniform(0.01, 0.15)   # planet/star radius ratio
        a_rs   = np.random.uniform(5, 50)        # semi-major axis ratio
        
        flux_with_transit = inject_transit(time, flux, period, t0, rp_rs, a_rs)
        
        save_file = f'dataset_synthetic/synthetic_{i:05d}.npz'
        np.savez(
            save_file,
            time=time,
            flux=flux_with_transit
        )
        synthetic_labels.append({
            'file': f'synthetic_{i:05d}.npz',
            'label': 1,
            'period': period,
            'rp_rs': rp_rs
        })
        
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{num_samples} samples.")

    print("Synthetic data generation complete!")

if __name__ == "__main__":
    generate_synthetics()
