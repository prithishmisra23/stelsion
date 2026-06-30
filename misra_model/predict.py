# misra_model/predict.py

import numpy as np
import torch
import lightkurve as lk

from .config import CONFIG
from .model import TechTonicsExoplanetDetector
from .preprocessor import preprocess
from .period_finder import find_best_period
from .view_generator import generate_all_views


def predict_star(kic_id, model_path=None, config=CONFIG):
    """
    Full end-to-end prediction for a single Kepler star.
    
    Input: KIC ID (e.g. 11442793 = Kepler-90)
    Output: Complete prediction report
    
    This is what you demo to ISRO judges.
    """
    if model_path is None:
        model_path = config["model_save_path"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = TechTonicsExoplanetDetector()
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded. Best training F1: {checkpoint.get('best_f1', 'N/A')}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {model_path}. Using uninitialized weights.")
    
    model = model.to(device)
    
    # Download from NASA MAST
    print(f"Downloading KIC {kic_id} from NASA MAST...")
    search = lk.search_lightcurve(f'KIC {kic_id}', mission='Kepler', cadence='long')
    
    if len(search) == 0:
        return {"error": f"No Kepler data found for KIC {kic_id}"}
    
    # Download and stitch all quarters
    lc_collection = search.download_all()
    lc = lc_collection.stitch().remove_nans()
    
    time = lc.time.value.astype(np.float32)
    flux = lc.flux.value.astype(np.float32)
    
    # Preprocess
    print("Preprocessing...")
    time_clean, flux_clean = preprocess(time, flux, method='adaptive')
    
    # Find period
    print("Running BLS period search...")
    period_info = find_best_period(time_clean, flux_clean, config)
    print(f"  Best period: {period_info['period']:.4f} days")
    print(f"  BLS power:   {period_info['bls_power']:.2f}")
    
    # Generate views
    global_v, local_v, matrix_2d = generate_all_views(
        time_clean, flux_clean, period_info, config
    )
    
    # Convert to tensors
    gv_t  = torch.tensor(global_v).unsqueeze(0).unsqueeze(0).to(device)
    lv_t  = torch.tensor(local_v).unsqueeze(0).unsqueeze(0).to(device)
    m2d_t = torch.tensor(matrix_2d).unsqueeze(0).unsqueeze(0).to(device)
    meta_t = torch.tensor([[1.0, 0.0, 0.0]]).to(device)
    
    # Predict with uncertainty
    print("Running neural network prediction...")
    mean_prob, uncertainty = model.predict_with_uncertainty(
        gv_t, lv_t, m2d_t, meta_t,
        n_samples=config["mc_dropout_samples"]
    )
    
    # Confidence level
    if uncertainty < 0.05:
        confidence = "HIGH"
    elif uncertainty < 0.15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW — FLAG FOR REVIEW"
    
    # Final classification
    is_planet = mean_prob > 0.5
    
    result = {
        'kic_id': kic_id,
        'is_exoplanet': is_planet,
        'probability': round(mean_prob, 4),
        'uncertainty': round(uncertainty, 4),
        'confidence': confidence,
        'period_days': round(period_info['period'], 4),
        'transit_duration_days': round(period_info['duration'], 4),
        'transit_depth': round(period_info['depth'], 6),
        'bls_power': round(period_info['bls_power'], 2),
        'verdict': 'EXOPLANET CANDIDATE' if is_planet else 'FALSE POSITIVE'
    }
    
    # Print report
    print("\n" + "="*50)
    print("  TEAM TECHTONICS — PREDICTION REPORT")
    print("="*50)
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("="*50)
    
    return result
