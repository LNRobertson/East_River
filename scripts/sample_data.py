# filepath: scripts/generate_sample_data.py
import pandas as pd

INPUT      = r"data\training\east_river_training-v2.h5"
OUTPUT_H5  = r"data\sample\east_river_training_sample-training-v2.h5"
OUTPUT_CSV = r"data\sample\east_river_training_sample-training-v2.csv"
KEY        = "df"
N          = 1000      # or whatever size you want

# 1) load just N rows instead of the whole DF
df = pd.read_hdf(INPUT, key=KEY, stop=N)

# 2) if you still want randomness (optional), sample from those N rows
# sample = df.sample(n=N, random_state=42)

# 3a) save back to HDF5
df.to_hdf(OUTPUT_H5, key=KEY, mode="w")
# 3b) optionally also save a CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote {len(df)} rows to:\n  - {OUTPUT_H5}\n  - {OUTPUT_CSV}")