import os
import sys
import numpy as np
import tensorflow as tf
import lightkurve as lk

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.filters import handle_missing_values
from model import build_secondary_model

def download_and_prepare_kepler_data():
    print("\n[Step 1] Loading Original Kepler Data via Lightkurve...")
    
    exoplanet_targets = ["Kepler-90", "Kepler-11", "Kepler-22", "Kepler-62", "Kepler-186"]
    quiet_targets = ["KIC 8462852", "KIC 11253772", "KIC 3733346", "KIC 11152511", "KIC 12158940"]
    
    x_raw = []
    y_raw = []
    
    segment_size = 2000
    
    def process_target(target, label):
        try:
            print(f" - Searching and downloading: {target}...")
            search_result = lk.search_lightcurve(target, quarter=4, author='Kepler')
            if len(search_result) == 0:
                search_result = lk.search_lightcurve(target, mission='Kepler')
            if len(search_result) == 0:
                raise ValueError(f"No light curves found for target {target}")
            
            lc = search_result[0].download()
            flux = np.array(lc.flux.value if hasattr(lc.flux, 'value') else lc.flux, dtype=float)
            
            # Clean missing values first so segmentation doesn't fail on NaNs
            flux = handle_missing_values(flux, method='interpolate')
            
            num_segments = len(flux) // segment_size
            if num_segments == 0:
                # Pad to at least one segment size if it is too short
                padded_flux = np.pad(flux, (0, segment_size - len(flux)), mode='edge')
                x_raw.append(padded_flux.tolist())
                y_raw.append(label)
                print(f"   -> Added 1 segment (padded) for {target}")
            else:
                for i in range(num_segments):
                    seg = flux[i * segment_size : (i + 1) * segment_size]
                    x_raw.append(seg.tolist())
                    y_raw.append(label)
                print(f"   -> Added {num_segments} segments of size {segment_size} for {target}")
        except Exception as e:
            print(f"   [Warning] Failed to fetch {target}: {e}. Generating simulated fallback...")
            # Fallback simulation
            sim_len = 6000
            time = np.arange(sim_len)
            noise = np.random.normal(0, 0.005, sim_len)
            transit = np.zeros(sim_len)
            if label == 1:
                # Add a couple of transit events
                transit[1000:1080] = -0.02
                transit[4000:4080] = -0.02
            mock_flux = 1.0 + noise + transit
            for i in range(sim_len // segment_size):
                seg = mock_flux[i * segment_size : (i + 1) * segment_size]
                x_raw.append(seg.tolist())
                y_raw.append(label)
            print(f"   -> Added 3 simulated fallback segments for {target}")

    # Process positive targets (confirmed planets)
    for target in exoplanet_targets:
        process_target(target, 1)
        
    # Process negative targets (quiet / non-transit stars)
    for target in quiet_targets:
        process_target(target, 0)
        
    print(f"Total segments prepared: {len(x_raw)}")
    return x_raw, y_raw

def train_and_evaluate():
    print("=" * 70)
    print("         SECONDARY Conv2D MODEL TRAINING (REAL KEPLER DATA)          ")
    print("=" * 70)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    x_raw, y_raw = download_and_prepare_kepler_data()
    
    print("\n[Step 2] Running Preprocessing Pipeline (padding to 7200 & augmenting)...")
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
        'segment_length': 2000, # target length of 2000 to reshape to 50x40
        'enable_augmentation': True, # Enable augmentation to reduce overfitting
        'test_size': 0.25,
        'val_size': 0.15,
        'random_state': 42
    })
    
    split_data = pipeline.prepare_dataset(x_raw, y_raw)
    train_x, train_y = split_data['train']
    val_x, val_y = split_data['val']
    test_x, test_y = split_data['test']
    
    # Reshape input to 2D (50, 40, 1)
    train_x_reshaped = train_x.reshape(-1, 50, 40, 1)
    val_x_reshaped = val_x.reshape(-1, 50, 40, 1)
    test_x_reshaped = test_x.reshape(-1, 50, 40, 1)
    
    print(f"Data shapes after reshaping to (50, 40, 1):")
    print(f" - Train: {train_x_reshaped.shape}")
    print(f" - Validation: {val_x_reshaped.shape}")
    print(f" - Test: {test_x_reshaped.shape}")
    
    # 3. Instantiate and compile secondary model
    print("\n[Step 3] Compiling Regularized Secondary Conv2D Model...")
    model = build_secondary_model(input_shape=(50, 40, 1), l2_reg=1e-3, dropout_rate_fc1=0.5, dropout_rate_fc2=0.3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()
    
    # 4. Train Model with Callbacks
    print("\n[Step 4] Training Secondary Model with Early Stopping & LR Decay...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=8, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_x_reshaped, train_y,
        validation_data=(val_x_reshaped, val_y),
        epochs=40,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluate on Train, Val and Test sets
    print("\n[Step 5] Evaluating Model Performance...")
    train_loss, train_acc = model.evaluate(train_x_reshaped, train_y, verbose=0)
    val_loss, val_acc = model.evaluate(val_x_reshaped, val_y, verbose=0)
    test_loss, test_acc = model.evaluate(test_x_reshaped, test_y, verbose=0)
    
    print("=" * 70)
    print("RESULTS:")
    print(f"Train Loss:      {train_loss:.4f} | Train Accuracy:      {train_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Loss:       {test_loss:.4f} | Test Accuracy:       {test_acc*100:.2f}%")
    print("=" * 70)

    # 6. Save the trained secondary model
    print("\n[Step 6] Saving Trained Secondary Model...")
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models'))
    os.makedirs(save_dir, exist_ok=True)
    
    weights_path = os.path.join(save_dir, 'secondary_model.weights.h5')
    model.save_weights(weights_path)
    print(f"Saved weights to: {weights_path}")
    
    model_path = os.path.join(save_dir, 'secondary_model.h5')
    model.save(model_path)
    print(f"Saved full model to: {model_path}")

if __name__ == '__main__':
    train_and_evaluate()

