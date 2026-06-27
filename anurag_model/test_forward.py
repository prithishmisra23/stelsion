import sys
import os
import tensorflow as tf
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.architecture import UpgradedExoplanetDetectorNet

def run_forward_pass_test():
    print("--- Starting SOTA 4-Input Forward Pass Validation ---")
    
    # 1. Initialize the upgraded model
    print("Initializing UpgradedExoplanetDetectorNet...")
    model = UpgradedExoplanetDetectorNet(input_len=2000, dropout=0.3, num_heads=4)
    print("Model initialized successfully.")
    
    # 2. Generate dummy 4-input tensors
    batch_size = 4
    global_len = 2000
    local_len = 200
    
    print(f"Creating dummy global input of shape (Batch={batch_size}, Len={global_len}, Channels=1)...")
    dummy_global = tf.random.normal((batch_size, global_len, 1))
    
    print(f"Creating dummy local input of shape (Batch={batch_size}, Len={local_len}, Channels=1)...")
    dummy_local = tf.random.normal((batch_size, local_len, 1))
    
    print(f"Creating dummy centroid input of shape (Batch={batch_size}, Len={global_len}, Channels=2)...")
    dummy_centroid = tf.random.normal((batch_size, global_len, 2))
    
    print(f"Creating dummy metadata input of shape (Batch={batch_size}, Features=3)...")
    dummy_metadata = tf.random.normal((batch_size, 3))
    
    inputs_dict = {
        'global': dummy_global,
        'local': dummy_local,
        'centroid': dummy_centroid,
        'metadata': dummy_metadata
    }
    
    # 3. Perform forward pass
    print("Running forward pass with 4 fused inputs...")
    preds, attn = model(inputs_dict, training=False)
        
    print("Forward pass completed successfully.")
    
    # 4. Check shapes and bounds
    print(f"Prediction output shape: {preds.shape} (Expected: ({batch_size}, 1))")
    print(f"Attention matrix shape: {attn.shape} (Expected: ({batch_size}, 63, 63))")
    
    # Convert values for checking
    preds_np = preds.numpy()
    min_pred = float(np.min(preds_np))
    max_pred = float(np.max(preds_np))
    print(f"Output probability range: [{min_pred:.4f}, {max_pred:.4f}]")
    
    # Assertions
    assert preds.shape == (batch_size, 1), f"Incorrect prediction shape: {preds.shape}"
    assert attn.shape == (batch_size, 63, 63), f"Incorrect attention shape: {attn.shape}"
    assert 0.0 <= min_pred <= 1.0, f"Probability out of range: {min_pred}"
    assert 0.0 <= max_pred <= 1.0, f"Probability out of range: {max_pred}"
    
    print("\n✓ Verification Successful! TensorFlow SOTA model compiles and runs with 4-branch fusion.")

if __name__ == "__main__":
    try:
        run_forward_pass_test()
    except Exception as e:
        print(f"\n✗ Test Failed: {str(e)}")
        sys.exit(1)
