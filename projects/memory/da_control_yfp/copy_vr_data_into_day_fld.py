"""
quantify licks and velocity during consolidation task
aug 2024
TODO: get first lick during probes
"""
#%%
import os, numpy as np, h5py, scipy, seaborn as sns, sys, pandas as pd, itertools
from pathlib import Path
import os
import re
import shutil
import pandas as pd
from datetime import datetime

# save to pdf
df = pd.read_csv(r"C:\Users\Han\Downloads\data_organization_halo_control_yfp_grab.csv", index_col = None)

# normalize mouse names (lowercase, strip spaces)
df["Animal"] = df["Animal"].str.strip().str.lower()

# normalize dates to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


output_root  = r"Y:\yfp_grabda"
animals = ['e244','e245','e246']#,'e242','e243']
behavior_dir  = r"G:\VR_data"


# --- PROCESS BEHAVIOR FILES ---
for fname in os.listdir(behavior_dir):
   if not fname.endswith(".mat"):
      continue

   # match pattern: E245_19_Jun_2025_time(10_07_54).mat
   m = re.match(r"(E\d+)_([0-9]{1,2})_([A-Za-z]{3})_([0-9]{4})", fname)
   if not m:
      print(f"Skipping {fname} (bad format)")
      continue

   mouse, day_str, month_str, year_str = m.groups()
   mouse = mouse.lower()
   date_str = f"{day_str} {month_str} {year_str}"
   file_date = datetime.strptime(date_str, "%d %b %Y")

   # --- MATCH TO CSV ---
   match = df[(df["Animal"] == mouse) & (df["Date"] == file_date)]
   if match.empty:
      print(f"No match for {fname} ({mouse}, {file_date.date()})")
      continue
   day_number = int(match["Day"].iloc[0])

   # --- MAKE OUTPUT PATH ---
   out_dir = os.path.join(output_root, mouse, str(day_number), 'behavior','vr')
   os.makedirs(out_dir, exist_ok=True)

   # --- COPY FILE ---
   src = os.path.join(behavior_dir, fname)
   dst = os.path.join(out_dir, fname)
   shutil.copy2(src, dst)

   print(f"Copied {fname} → {out_dir}")