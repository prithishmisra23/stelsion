#this is a one time runner code , repeated running will cause repeated downloads
import os
import pandas as pd
from lightkurve import search_lightcurve
import numpy as np
import warnings

from misra_model.config import CONFIG
from misra_model.preprocessor import preprocess
from misra_model.period_finder import find_best_period

# Ignore lightkurve warnings about downloading only 1 file
warnings.filterwarnings('ignore', category=UserWarning)

koi = pd.read_csv("modified datasets/koi_cumulative_labeled.csv" , comment='#')

# Increase sample size from 10 to 50 per class (150 total stars) to beat Ramanuj
subset=(
    koi.groupby("signal_class", group_keys=False).head(50)
)
os.makedirs("dataset", exist_ok=True)

records=[]

for _,row in subset.iterrows():
    kepid = row["kepid"]
    label = row["signal_class"]
    print(f"Processing {kepid}...")
    try:
        search = search_lightcurve(f"KIC {kepid}", mission="kepler")
        if len(search) == 0:
            print(f"No data found for {kepid}")
            continue
        lc = search.download()
        if lc is None:
            continue
        lc = lc.remove_nans().normalize()

        time = lc.time.value
        flux = lc.flux.value
        
        # Precompute BLS Period here on the CPU ONCE so the GPU doesn't have to during training
        time_clean, flux_clean = preprocess(time, flux, method='adaptive')
        period_info = find_best_period(time_clean, flux_clean, CONFIG)

        np.savez(
            f"dataset/{kepid}.npz",
            time=time,
            flux=flux,
            period=period_info['period'],
            t0=period_info['t0'],
            duration=period_info['duration']
        )

        records.append({
            "kepid": kepid,
            "label": label,
            "file": f"dataset/{kepid}.npz"
        })

        print(f"Successfully saved {kepid} [Period: {period_info['period']:.2f} days]")
        
    except Exception as e:
        print(f"Failed on {kepid}: {e}")

pd.DataFrame(records).to_csv("dataset_index.csv", index=False)
