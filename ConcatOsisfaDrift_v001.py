
#%%
import xarray as xr
import os
import pandas as pd
import numpy as np
from netCDF4 import num2date

base_dir = "/home/waynedj/Data/seaicedrift/osisaf/24hr/south/merged/"

for year in range(2019, 2020):
    year = str(year)
    print(f"Processing year: {year}")

    year_dir = os.path.join(base_dir, year)

    filelist = []
    for month in [f"{m:02d}" for m in range(1, 13)]:
        month_dir = os.path.join(year_dir, month)
        if os.path.exists(month_dir):
            for fname in sorted(os.listdir(month_dir)):
                if fname.endswith(".nc"):
                    filelist.append(os.path.join(month_dir, fname))

    if not filelist:
        print(f"No files found for {year}, skipping...")
        continue

    # Open files without decoding times first
    ds = xr.open_mfdataset(
        filelist,
        combine="by_coords",
        decode_times=False,   # <-- prevent time decoding problems
        parallel=True,
        data_vars=["dX", "dY"],
    )

    # Ensure sorted order
    ds = ds.sortby("time")

    out_fname = f"ice_drift_sh_ease2-750_cdr-v1p0_24h-{year}_v001.nc"
    out_path = os.path.join(year_dir, out_fname)
    ds.to_netcdf(out_path)

    print(f"Saved yearly dataset: {out_path}")




















# %%
