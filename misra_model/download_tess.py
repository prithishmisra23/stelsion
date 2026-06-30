import lightkurve as lk
import numpy as np
import os
import pandas as pd
import sys

# Download the TOI catalog if not present (handled dynamically if the CSV is missing)
catalog_file = 'toi_catalog.csv'
if not os.path.exists(catalog_file):
    print("Downloading TESS TOI catalog...")
    try:
        toi = pd.read_csv('https://exofop.ipac.caltech.edu/tess/download_toi.php')
        toi.to_csv(catalog_file, index=False)
    except Exception as e:
        print(f"Error fetching catalog: {e}")
        sys.exit(1)
else:
    toi = pd.read_csv(catalog_file)

os.makedirs('dataset_tess', exist_ok=True)

print("Starting TESS short-cadence downloads...")
# Download the first 200 items as specified
for _, row in toi.head(200).iterrows():
    tic_id = int(row['TIC ID'])
    save_path = f'dataset_tess/{tic_id}.npz'
    
    if os.path.exists(save_path):
        print(f"Skipping TIC {tic_id} (already downloaded).")
        continue
        
    try:
        search = lk.search_lightcurve(
            f'TIC {tic_id}',
            mission='TESS',
            cadence='short'
        )
        if len(search) == 0:
            print(f"No short cadence data found for TIC {tic_id}")
            continue
            
        lc = search[0].download().remove_nans().normalize()
        np.savez(
            save_path,
            time=lc.time.value,
            flux=lc.flux.value
        )
        print(f"Successfully saved TIC {tic_id}")
    except Exception as e:
        print(f"Failed TIC {tic_id}: {e}")
