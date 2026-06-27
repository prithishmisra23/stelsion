import os
import sys
import numpy as np
import tensorflow as tf
import batman

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.pipeline import DualViewPipeline

def generate_mandel_agol_transit(length, period, rp, t0):
    """
    Generates a high-fidelity Mandel & Agol (2002) limb-darkened transit profile
    using the batman-package.
    rp: planet-to-star radius ratio
    """
    time = np.linspace(0, 10, length)
    
    params = batman.TransitParams()
    params.t0 = (t0 / length) * 10.0  # time of inferior conjunction
    params.per = (period / length) * 10.0 # orbital period
    params.rp = rp                      # planet radius (in units of stellar radii)
    params.a = 15.                      # semi-major axis (in units of stellar radii)
    params.inc = 89.5                   # orbital inclination (in degrees)
    params.ecc = 0.                     # eccentricity
    params.w = 90.                      # longitude of periastron (in degrees)
    params.u = [0.1, 0.3]               # limb darkening coefficients
    params.limb_dark = "quadratic"      # limb darkening model
    
    m = batman.TransitModel(params, time)
    flux = m.light_curve(params)
    
    # Batman generates 1.0 for out of transit, and < 1.0 for in-transit. 
    # We want a dip profile that is 0 for out of transit and < 0 for in-transit.
    return flux - 1.0

class ExoplanetDataset(tf.keras.utils.Sequence):
    def __init__(self, num_samples=160, batch_size=16, length=2000, inject_prob=0.5, **kwargs):
        """
        SOTA Keras Sequence Dataset with:
        1. Mandel-Agol Limb-Darkened Transits.
        2. Centroid Motion Simulation (X/Y) to simulate eclipsing binaries.
        3. Tabular Stellar Metadata (Radius, Mass, Teff).
        """
        super(ExoplanetDataset, self).__init__(**kwargs)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.length = length
        self.inject_prob = inject_prob
        self.pipeline = DualViewPipeline()
        
        self.baselines = []
        for _ in range(num_samples):
            time = np.linspace(0, 10, length)
            p1 = np.random.uniform(2.0, 5.0)
            p2 = np.random.uniform(0.3, 1.0)
            amp1 = np.random.uniform(0.01, 0.03)
            amp2 = np.random.uniform(0.002, 0.008)
            stellar_var = amp1 * np.sin(2 * np.pi * time / p1) + amp2 * np.cos(2 * np.pi * time / p2)
            
            jitter = np.zeros(length)
            if np.random.random() < 0.3:
                jump_idx = np.random.randint(200, length - 200)
                jitter[jump_idx:] += np.random.uniform(-0.006, 0.006)
                
            flares = np.zeros(length)
            if np.random.random() < 0.2:
                for _ in range(np.random.randint(1, 3)):
                    flare_idx = np.random.randint(100, length - 100)
                    flare_amp = np.random.uniform(0.01, 0.03)
                    width = np.random.uniform(2, 6)
                    flares += flare_amp * np.exp(-0.5 * ((np.arange(length) - flare_idx) / width) ** 2)
                    
            noise = np.random.normal(0, 0.003, length)
            self.baselines.append(1.0 + stellar_var + jitter + flares + noise)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        global_batch = []
        local_batch = []
        centroid_batch = []
        meta_batch = []
        label_batch = []
        
        for i in range(start_idx, end_idx):
            flux = self.baselines[i].copy()
            centroid_x = np.random.normal(0, 0.001, self.length)
            centroid_y = np.random.normal(0, 0.001, self.length)
            
            # Metadata: Radius (R_sun), Mass (M_sun), Teff (K / 1000)
            r_star = np.random.uniform(0.1, 2.0)
            m_star = np.random.uniform(0.1, 2.0)
            teff = np.random.uniform(3.0, 7.0)
            metadata = [r_star, m_star, teff]
            
            label = 0.0
            is_planet = False
            is_beb = False # Background Eclipsing Binary (False Positive)
            
            rand_val = np.random.random()
            if rand_val < self.inject_prob:
                # Decide if it's a true planet or a false positive (BEB)
                if np.random.random() < 0.7:
                    is_planet = True
                    label = 1.0
                else:
                    is_beb = True
                    label = 0.0
                
                period = np.random.uniform(300, 600)
                rp = np.random.uniform(0.05, 0.2) # Planet radius ratio
                t0 = np.random.uniform(50, 250)
                
                transit_dip = generate_mandel_agol_transit(self.length, period, rp, t0)
                flux += transit_dip
                
                # If it's a BEB, shift the centroid precisely during the transit!
                if is_beb:
                    dip_mask = transit_dip < -0.001
                    centroid_shift_x = np.random.uniform(0.01, 0.05)
                    centroid_shift_y = np.random.uniform(0.01, 0.05)
                    centroid_x[dip_mask] += centroid_shift_x
                    centroid_y[dip_mask] += centroid_shift_y
                
            flux += np.random.normal(0, 0.001, self.length)
            
            global_view, local_view, _, _ = self.pipeline.process(flux)
            centroid_combined = np.stack([centroid_x, centroid_y], axis=-1)
            
            global_batch.append(global_view[:, np.newaxis])
            local_batch.append(local_view[:, np.newaxis])
            centroid_batch.append(centroid_combined)
            meta_batch.append(metadata)
            label_batch.append([label])
            
        inputs_dict = {
            'global': np.array(global_batch, dtype=np.float32),
            'local': np.array(local_batch, dtype=np.float32),
            'centroid': np.array(centroid_batch, dtype=np.float32),
            'metadata': np.array(meta_batch, dtype=np.float32)
        }
        
        return inputs_dict, np.array(label_batch, dtype=np.float32)

if __name__ == "__main__":
    print("Testing SOTA ExoplanetDataset...")
    dataset = ExoplanetDataset(num_samples=10, batch_size=2, inject_prob=0.5)
    inputs, y = dataset[0]
    print(f"Dataset batch test:")
    print(f"Global View Batch shape: {inputs['global'].shape}")
    print(f"Local View Batch shape: {inputs['local'].shape}")
    print(f"Centroid Batch shape: {inputs['centroid'].shape}")
    print(f"Metadata Batch shape: {inputs['metadata'].shape}")
    print(f"Label Batch shape: {y.shape}")
    print("\n✓ Dataset test passed successfully!")
