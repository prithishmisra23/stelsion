import os
import sys
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Ensure current directory is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing.synthetic import generate_synthetic_transit
from preprocessing.pipeline import PreprocessingPipeline
from models.architecture import ExoplanetDetectorNet
from training.train import Trainer
from evaluation.explainability import (
    GradCAM1D,
    estimate_uncertainty_mc_dropout,
    estimate_transit_parameters,
    analyze_false_positives
)

def build_and_run_pipeline():
    print("=" * 70)
    print("         AstroAI: EXOPLANET DETECTION ML PIPELINE RUNNER          ")
    print("=" * 70)
    
    # 1. Generate Synthetic Dataset
    print("\n[Step 1/5] Generating Synthetic Dataset...")
    np.random.seed(42)
    tf.random.set_seed(42)
    
    x_raw = []
    y_raw = []
    candidates = []
    
    # Generate 30 transit curves (exoplanets)
    print(" - Generating 30 Exoplanet transit light curves...")
    for i in range(30):
        # Vary depth and period slightly
        depth = np.random.uniform(0.015, 0.035)
        period = np.random.uniform(400, 600)
        curve_data = generate_synthetic_transit(
            seq_len=2000,
            has_transit=True,
            depth=depth,
            period=period,
            duration=80.0,
            noise_level=0.01,
            stellar_var_amp=0.02,
            seed=i
        )
        x_raw.append(curve_data["flux"])
        y_raw.append(1)
        candidates.append(f"ExoCandidate-{i+1:02d}")
        
    # Generate 30 non-transit curves (no planets/noise/stellar variability)
    print(" - Generating 30 Non-Transit light curves (eclipsing binary, stellar noise)...")
    for i in range(30):
        # 10 pure noise / variability
        if i < 10:
            curve_data = generate_synthetic_transit(
                seq_len=2000,
                has_transit=False,
                noise_level=0.012,
                stellar_var_amp=0.03,
                seed=100+i
            )
            x_raw.append(curve_data["flux"])
            y_raw.append(0)
            candidates.append(f"StellarNoise-{i+1:02d}")
        # 10 eclipsing binary-like (very deep periodic dips)
        elif i < 20:
            curve_data = generate_synthetic_transit(
                seq_len=2000,
                has_transit=True,
                depth=0.15,  # EB has huge depth
                period=450,
                duration=120.0,
                noise_level=0.008,
                stellar_var_amp=0.01,
                seed=200+i
            )
            x_raw.append(curve_data["flux"])
            y_raw.append(0)  # Label as 0 (false positive exoplanet target)
            candidates.append(f"BinaryEclipse-{i-9:02d}")
        # 10 single-point glitches
        else:
            curve_data = generate_synthetic_transit(
                seq_len=2000,
                has_transit=False,
                noise_level=0.01,
                stellar_var_amp=0.01,
                seed=300+i
            )
            # Inject a sudden glitch spike/dip
            flux = np.array(curve_data["flux"])
            glitch_idx = np.random.randint(200, 1800)
            flux[glitch_idx] -= 0.12 # single deep dip
            x_raw.append(flux.tolist())
            y_raw.append(0)
            candidates.append(f"GlitchTarget-{i-19:02d}")
            
    print(f"Generated a total of {len(x_raw)} light curves.")
    
    # 2. Preprocess & Split Data
    print("\n[Step 2/5] Running Preprocessing Pipeline...")
    pipeline = PreprocessingPipeline({
        'missing_value_method': 'interpolate',
        'normalization_method': 'median',
        'sigma_clipping_sigma': 3.0,
        'sigma_clipping_iters': 2,
        'wavelet_type': 'db4',
        'wavelet_level': 2,
        'sg_window': 15,
        'sg_polyorder': 2,
        'median_kernel': 5,
        'stellar_var_window': 101,
        'segment_length': 2000,
        'enable_augmentation': False,  # disable augmentation here to keep training quick and deterministic
        'test_size': 0.25,
        'val_size': 0.15,
        'random_state': 42
    })
    
    split_data = pipeline.prepare_dataset(x_raw, y_raw)
    train_x, train_y = split_data['train']
    val_x, val_y = split_data['val']
    test_x, test_y = split_data['test']
    
    print(f"Dataset split completed:")
    print(f" - Train samples: {len(train_x)}")
    print(f" - Validation samples: {len(val_x)}")
    print(f" - Test samples: {len(test_x)}")
    
    # 3. Train Model
    print("\n[Step 3/5] Instantiating and Training 1D CNN + Self-Attention Model...")
    model = ExoplanetDetectorNet(input_len=2000)
    trainer = Trainer(model=model, lr=0.001, checkpoint_dir='saved_models')
    
    # Train for a small number of epochs to demonstrate function quickly
    print("Training for 6 epochs...")
    history = trainer.train(
        (train_x, train_y),
        (val_x, val_y),
        epochs=6,
        batch_size=16,
        early_stopping_patience=3
    )
    
    # Load best checkpoint
    best_model_path = os.path.join('saved_models', 'best_model.weights.h5')
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint model from: {best_model_path}")
        trainer.load_checkpoint(best_model_path)
    
    # 4. Predict on Test Set
    print("\n[Step 4/5] Running Inference on Test Set...")
    
    # Header for results
    print("-" * 125)
    print(f"{'Target ID':<20} | {'True':<5} | {'Prob':<6} | {'Unc.':<6} | {'Rel.':<6} | {'Depth%':<6} | {'Dur(h)':<6} | {'Period':<6} | {'Verdict':<28} | {'Reason'}")
    print("-" * 125)
    
    correct_predictions = 0
    
    # We will choose one exoplanet candidate to plot explainability map
    sample_to_plot_idx = None
    
    for idx in range(len(test_x)):
        flux = test_x[idx]
        true_label = int(test_y[idx])
        target_name = f"TestTarget-{idx+1:02d}"
        
        # Format input tensor [1, 2000, 1]
        input_tensor = np.array(flux, dtype=np.float32)[np.newaxis, :, np.newaxis]
        
        # MC Dropout Uncertainty
        mean_prob, uncertainty, reliability = estimate_uncertainty_mc_dropout(model, input_tensor, num_samples=10)
        
        # Estimate parameters
        params = estimate_transit_parameters(flux)
        
        # False Positive Analysis
        fp_analysis = analyze_false_positives(flux)
        
        # Verdict calculation
        pred_label = 1 if mean_prob > 0.5 else 0
        if pred_label == true_label:
            correct_predictions += 1
            
        # Select first exoplanet target to plot details
        if true_label == 1 and sample_to_plot_idx is None:
            sample_to_plot_idx = idx
            
        print(f"{target_name:<20} | {true_label:<5} | {mean_prob:.2f} | {uncertainty:.2f} | {reliability:<6} | {params['depth_percent']:5.2f}% | {params['duration_hours']:5.2f} | {params['period_days']:5.2f} | {fp_analysis['verdict']:<28} | {fp_analysis['reason']}")
        
    test_accuracy = correct_predictions / len(test_x)
    print("-" * 125)
    print(f"Overall Test Accuracy: {test_accuracy*100:.2f}% ({correct_predictions}/{len(test_x)} correct)")
    print("-" * 125)
    
    # 5. Generate and Save Explanation Plot
    if sample_to_plot_idx is not None:
        print("\n[Step 5/5] Generating Visual Explainability Plots (Grad-CAM & Attention)...")
        os.makedirs("results", exist_ok=True)
        
        sample_flux = test_x[sample_to_plot_idx]
        input_tensor = np.array(sample_flux, dtype=np.float32)[np.newaxis, :, np.newaxis]
        
        # GradCAM
        gradcam = GradCAM1D(model, model.res3.conv2)
        heatmap = gradcam.generate_heatmap(input_tensor)
        
        # Attention
        _, attn_map = model(input_tensor, training=False)
        attn_list = []
        if attn_map is not None:
            attn_np = attn_map.numpy()[0]
            attn_list = np.mean(attn_np, axis=0)
            
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 1. Flux Plot
        axes[0].plot(sample_flux, color='#1f77b4', label='Denoised Normalized Flux', linewidth=1.5)
        axes[0].set_title('Exoplanet Candidate Light Curve (Test Target)', fontsize=14, color='#333333')
        axes[0].set_ylabel('Normalized Flux', fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend()
        
        # 2. Grad-CAM Activation Heatmap
        x_vals = np.arange(len(sample_flux))
        axes[1].fill_between(x_vals, 0, heatmap, color='orange', alpha=0.7, label='1D Grad-CAM Activation')
        axes[1].set_ylabel('Activation Strength', fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()
        
        # 3. Attention Weights
        if len(attn_list) > 0:
            # Interpolate attention map to original length if needed (attn output size depends on poolings)
            attn_interp = np.interp(np.linspace(0, len(attn_list), len(sample_flux)), np.arange(len(attn_list)), attn_list)
            axes[2].plot(attn_interp, color='green', label='Attention Map (Temporal Correlation)', linewidth=1.5)
            axes[2].set_ylabel('Attention Weight', fontsize=12)
            axes[2].grid(True, linestyle='--', alpha=0.5)
            axes[2].legend()
            
        plt.xlabel('Time Step (Observations)', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join("results", "explainability_report.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Visual explainability plot saved successfully to: results/explainability_report.png")
        plt.close()
        
    print("\nML Pipeline successfully executed! All CNN layers run and predict properly.")
    print("=" * 70)

if __name__ == "__main__":
    build_and_run_pipeline()
