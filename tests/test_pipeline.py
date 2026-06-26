import numpy as np
import tensorflow as tf
from preprocessing.filters import handle_missing_values, normalize_flux, remove_outliers_sigma_clipping
from models.architecture import ExoplanetDetectorNet

def test_missing_values():
    flux = np.array([1.0, 2.0, np.nan, 4.0])
    fixed_flux = handle_missing_values(flux, method='interpolate')
    assert not np.any(np.isnan(fixed_flux))
    assert fixed_flux[2] == 3.0

def test_normalization():
    flux = np.array([10.0, 10.0, 10.0, 10.0])
    norm_flux = normalize_flux(flux, method='median')
    assert np.all(norm_flux == 0.0)

def test_sigma_clipping():
    flux = np.array([1.0, 1.1, 1.0, 10.0, 1.0, 0.9])
    clipped = remove_outliers_sigma_clipping(flux, sigma=2.0)
    assert clipped[3] < 10.0 # Outlier is reduced

def test_model_forward():
    model = ExoplanetDetectorNet(input_len=2000)
    model(np.zeros((1, 2000, 1), dtype=np.float32), training=False)
    
    # Batch size 2, sequence length 2000, 1 channel
    dummy_input = tf.random.normal((2, 2000, 1))
    out, attn = model(dummy_input, training=False)
        
    assert out.shape == (2, 1)
    assert np.min(out) >= 0.0
    assert np.max(out) <= 1.0
    assert attn is not None

def test_noise_estimation_and_adaptive():
    from preprocessing.filters import estimate_noise
    from preprocessing.pipeline import PreprocessingPipeline
    
    flux_low_noise = np.random.normal(1.0, 0.001, 2000)
    flux_high_noise = np.random.normal(1.0, 0.03, 2000)
    
    assert estimate_noise(flux_low_noise) < 0.005
    assert estimate_noise(flux_high_noise) > 0.015
    
    pipeline = PreprocessingPipeline()
    res_low = pipeline.process_single_curve(flux_low_noise)
    res_high = pipeline.process_single_curve(flux_high_noise)
    
    assert len(res_low) == 2000
    assert len(res_high) == 2000

def test_synthetic_generator():
    from preprocessing.synthetic import generate_synthetic_transit
    
    res = generate_synthetic_transit(seq_len=1000, has_transit=True, noise_level=0.01)
    assert len(res["flux"]) == 1000
    assert res["has_transit"] is True
    assert res["depth"] == 0.02

def test_mc_dropout_and_params():
    from evaluation.explainability import estimate_uncertainty_mc_dropout, estimate_transit_parameters
    
    model = ExoplanetDetectorNet(input_len=2000)
    model(np.zeros((1, 2000, 1), dtype=np.float32), training=False)
    dummy_input = tf.random.normal((1, 2000, 1))
    
    mean_prob, uncertainty, reliability = estimate_uncertainty_mc_dropout(model, dummy_input, num_samples=5)
    assert 0.0 <= mean_prob <= 1.0
    assert uncertainty >= 0.0
    assert reliability in ["Low", "Medium", "High"]
    
    flux = np.ones(2000)
    flux[400:450] = 0.95  # simulated dip
    params = estimate_transit_parameters(flux)
    assert params["depth_percent"] > 0.0
    assert params["duration_hours"] > 0.0
