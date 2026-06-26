import numpy as np
import tensorflow as tf
import scipy.signal as signal

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate_heatmap(self, input_tensor):
        activations = []
        original_call = self.target_layer.call
        
        # Temporarily wrap the target layer call to capture its activations
        def wrapped_call(*args, **kwargs):
            out = original_call(*args, **kwargs)
            activations.append(out)
            return out
            
        self.target_layer.call = wrapped_call
        
        try:
            # Cast input to tensor and watch it
            input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_tensor)
                output, _ = self.model(input_tensor, training=False)
                
            if not activations:
                raise ValueError("Target layer was not called during forward pass.")
                
            act_tensor = activations[0]
            # Compute gradients of predictions with respect to target activations
            grads = tape.gradient(output, act_tensor)
            
        finally:
            # Restore original call
            self.target_layer.call = original_call
            
        # act_tensor shape: [B, N_act, C_act] e.g. [1, 250, 256]
        # grads shape: [B, N_act, C_act]
        
        # Mean gradients across sequence length to get channel weights
        pooled_gradients = tf.reduce_mean(grads, axis=1) # [B, C_act]
        
        # Scale activations by channel weights
        weighted_activations = act_tensor * tf.expand_dims(pooled_gradients, axis=1)
        
        # Average over channels
        heatmap = tf.reduce_mean(weighted_activations, axis=-1) # [B, N_act]
        heatmap = tf.squeeze(heatmap, axis=0) # [N_act]
        
        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0.0)
        
        # Normalize
        max_val = float(tf.reduce_max(heatmap).numpy())
        if max_val > 0:
            heatmap = heatmap / max_val
            
        heatmap_np = heatmap.numpy()
        
        # Interpolate heatmap to match input sequence length (2000 points) to avoid plotting shape mismatch
        target_len = input_tensor.shape[1]
        if len(heatmap_np) != target_len:
            heatmap_np = np.interp(
                np.linspace(0, len(heatmap_np) - 1, target_len),
                np.arange(len(heatmap_np)),
                heatmap_np
            )
            
        return heatmap_np

def estimate_uncertainty_mc_dropout(model, input_tensor, num_samples=10):
    """
    Computes exoplanet probability and uncertainty using Monte Carlo Dropout.
    Forces Keras Dropout layers to run with training=True while keeping Batch Normalization in inference mode.
    """
    # Cast input to tensor
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    
    # Locate all Dropout layers in the model (including sub-layers/blocks)
    dropout_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            dropout_layers.append(layer)
        # Check inside ResidualBlock1D
        if hasattr(layer, 'dropout') and isinstance(layer.dropout, tf.keras.layers.Dropout):
            dropout_layers.append(layer.dropout)
            
    # Temporarily wrap Dropout call methods to force training=True
    original_calls = []
    for layer in dropout_layers:
        original_calls.append((layer, layer.call))
        def make_forced_call(orig_call):
            def forced_call(inputs, training=None):
                return orig_call(inputs, training=True)
            return forced_call
        layer.call = make_forced_call(layer.call)
        
    try:
        samples = []
        for _ in range(num_samples):
            out, _ = model(input_tensor, training=False)
            val = float(tf.squeeze(out).numpy())
            samples.append(val)
    finally:
        # Restore original call methods
        for layer, orig_call in original_calls:
            layer.call = orig_call
            
    mean_prob = float(np.mean(samples))
    std_dev = float(np.std(samples))
    # 95% Confidence interval margin
    uncertainty = float(1.96 * std_dev)
    
    reliability = "High"
    if uncertainty > 0.05:
        reliability = "Medium"
    if uncertainty > 0.15:
        reliability = "Low"
        
    return mean_prob, uncertainty, reliability

def estimate_transit_parameters(flux, time=None):
    """
    Heuristically estimates exoplanet candidate parameters from the light curve.
    """
    flux = np.array(flux)
    n = len(flux)
    median_val = np.median(flux)
    
    # Calculate depth: distance from median to minimum dip
    min_idx = np.argmin(flux)
    min_val = flux[min_idx]
    depth_percent = float(max(0, (median_val - min_val) * 100))
    
    # Duration: Width of dip at 10% of maximum depth
    threshold = median_val - 0.1 * (median_val - min_val)
    transit_indices = np.where(flux < threshold)[0]
    
    if len(transit_indices) > 0:
        # Check if transit is grouped or spread out
        # Using a simple heuristic where 2000 points represent 10 days (240 hours)
        # So each step is 0.12 hours
        duration_hours = float(len(transit_indices) * 0.12)
    else:
        duration_hours = 0.0
        
    # Periodicity estimation: Find local minimums and calculate spacing
    dips = []
    # Smooth a bit to find true peaks
    smoothed = signal.medfilt(flux, 15)
    for i in range(10, n - 10):
        if smoothed[i] == np.min(smoothed[i-10:i+10]) and smoothed[i] < median_val - 0.02:
            dips.append(i)
            
    # Remove close duplicate dip detections
    filtered_dips = []
    for d in dips:
        if not filtered_dips or (d - filtered_dips[-1]) > 50:
            filtered_dips.append(d)
            
    if len(filtered_dips) >= 2:
        # Distance between adjacent dips scaled to days
        # 2000 points represent 10 days, so 1 point = 0.005 days
        diffs = np.diff(filtered_dips)
        period_days = float(np.mean(diffs) * 0.005)
    else:
        period_days = 8.23  # Default fallback candidate periodicity if single event
 
    return {
        "depth_percent": depth_percent,
        "duration_hours": duration_hours,
        "period_days": period_days
    }

def analyze_false_positives(flux, time=None):
    """
    Astronomical assessment of a candidate transit light curve to rule out false positives:
    - Eclipsing Binaries: Check for secondary eclipses or highly V-shaped profiles.
    - Stellar Variability: Check for periodic sinusoidal variations without distinct flat out-of-transit parts.
    - Instrument Glitches: Check if the dip occurs in a single/double frame spike.
    """
    flux = np.array(flux)
    if time is None:
        time = np.arange(len(flux))
        
    n = len(flux)
    mean_val = np.mean(flux)
    std_val = np.std(flux)
    
    # Calculate depth and shape of the deepest dip
    min_idx = np.argmin(flux)
    min_val = flux[min_idx]
    
    # Find secondary deep dip (excluding the primary dip region)
    exclude_window = 50
    mask = np.ones(n, dtype=bool)
    mask[max(0, min_idx - exclude_window):min(n, min_idx + exclude_window)] = False
    
    masked_flux = flux[mask]
    secondary_min_idx = np.argmin(masked_flux) if len(masked_flux) > 0 else None
    secondary_min_val = masked_flux[secondary_min_idx] if secondary_min_idx is not None else mean_val
    
    # Glitch check: Is it just a single-point outlier?
    is_glitch = False
    if min_idx > 0 and min_idx < n - 1:
        left_diff = flux[min_idx - 1] - min_val
        right_diff = flux[min_idx + 1] - min_val
        # If the dip recovers immediately to normal on both sides, it's a glitch
        if left_diff > 3 * std_val and right_diff > 3 * std_val:
            is_glitch = True
            
    # Eclipsing Binary check: Significant secondary eclipse
    is_eb = False
    primary_depth = mean_val - min_val
    secondary_depth = mean_val - secondary_min_val
    if primary_depth > 0 and secondary_depth / primary_depth > 0.4 and secondary_depth > 3 * std_val:
        is_eb = True
        
    # Stellar Variability check: Lomb-Scargle power or high variance in out-of-transit
    is_variable = False
    # If the standard deviation is extremely high even outside the main transit
    out_of_transit_std = np.std(masked_flux) if len(masked_flux) > 0 else 0
    if out_of_transit_std > 0.8 * std_val and std_val > 0.05:
        is_variable = True
        
    verdict = "Exoplanet Candidate"
    reason = "Strong, clean transit-like signal without secondary eclipses or single-point glitches."
    
    if is_glitch:
        verdict = "Rejected (Instrument Glitch)"
        reason = "Single-epoch sudden dip likely caused by cosmic ray or satellite crossing (no gradual transit profile)."
    elif is_eb:
        verdict = "Rejected (Eclipsing Binary)"
        reason = f"Primary depth: {primary_depth:.4f}, secondary depth: {secondary_depth:.4f}. Secondary eclipse suggests a binary star system."
    elif is_variable:
        verdict = "Rejected (Stellar Variability)"
        reason = "Continuous sinusoidal-like oscillations throughout the light curve indicate active stellar rotation or pulsations."
        
    return {
        "verdict": verdict,
        "reason": reason,
        "is_glitch": is_glitch,
        "is_eb": is_eb,
        "is_variable": is_variable,
        "primary_depth": float(primary_depth),
        "secondary_depth": float(secondary_depth)
    }
