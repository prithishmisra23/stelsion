# misra_model/config.py
# Central configuration — change only here, affects everything

CONFIG = {
    # Data
    "dataset_dir": "dataset",           # where your .npz files live
    "catalog_path": "dataset_index.csv", # labels CSV
    "label_column": "signal_class",        # column name for labels
    
    # Preprocessing
    "sigma_clip": 3.0,
    "detrend_method": "adaptive",          # 'savgol', 'wavelet', or 'adaptive'
    "savgol_window": 301,
    "wavelet_type": "db4",
    "wavelet_level": 4,
    
    # BLS Period Search
    "period_min": 0.5,                     # days
    "period_max": 20.0,                    # days
    "n_periods": 500,
    
    # View Generation
    "global_bins": 2001,
    "local_bins": 201,
    "local_transit_widths": 4,             # show 4 transit durations either side
    "matrix_orbits": 10,
    "matrix_phase_bins": 200,
    
    # Model
    "dropout_rate": 0.3,
    "mc_dropout_samples": 20,
    
    # Training
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 100,
    "focal_alpha": 0.5,
    "focal_gamma": 2.0,
    "device": "cuda",                      # 'cuda' or 'cpu'
    
    # Output
    "model_save_path": "misra_model/best_model.pth",
    "results_dir": "misra_model/results/",
}
