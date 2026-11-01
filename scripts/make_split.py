# scripts/make_split.py
# pip install pandas scikit-learn
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "dataset")

SPOT_RADIUS_M = 4      # photos within this distance are the same "spot"; raise if GPS is jittery
VAL_FRAC = 0.2
SEED = 42

df = pd.read_csv(os.path.join(DATA, "metadata.csv"))

# --- 1. Cluster photos into spots -------------------------------------------
# Scale degrees to ~meters so DBSCAN's eps is isotropic: 1 deg lat ≈ 111 km,
# 1 deg lon ≈ 111 km * cos(lat).
lat0 = np.radians(df["Latitude"].mean())
xy = np.column_stack([
    df["Latitude"].to_numpy() * 111_000,
    df["Longitude"].to_numpy() * 111_000 * np.cos(lat0),
])
df["spot_id"] = DBSCAN(eps=SPOT_RADIUS_M, min_samples=1).fit_predict(xy)

per_spot = df.groupby("spot_id").size()
print(f"{len(df)} photos -> {len(per_spot)} spots "
      f"(min {per_spot.min()}, median {per_spot.median():.0f}, max {per_spot.max()} photos/spot)")

# --- 2. Hold out whole spots for validation --------------------------------
splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
tr_idx, va_idx = next(splitter.split(df, groups=df["spot_id"]))

train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
assert not set(train_df["spot_id"]) & set(val_df["spot_id"]), "spot leaked across split"

train_df.to_csv(os.path.join(DATA, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(DATA, "val_split.csv"), index=False)

print(f"train: {len(train_df):5d} photos / {train_df['spot_id'].nunique()} spots")
print(f"val:   {len(val_df):5d} photos / {val_df['spot_id'].nunique()} spots")