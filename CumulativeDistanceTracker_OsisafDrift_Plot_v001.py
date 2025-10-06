#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl

def CustomColorMap1():
    # Generate custom cmap
    boundaries = np.arange(0, 1600, 100)  # adjust max value as needed
    cmap_custom = mpl.colormaps.get_cmap("jet").resampled(len(boundaries))
    colors = list(cmap_custom(np.arange(len(boundaries))))
    cmap1 = mpl.colors.ListedColormap(colors, "k")
    cmap1.set_over(colors[-1])  # last color for values beyond vmax
    return cmap1

def CustomColorMap2():
    # Generate custom cmap
    boundaries = np.arange(1, 8, 1)  # adjust max value as needed
    cmap_custom = mpl.colormaps.get_cmap("viridis").resampled(len(boundaries))
    colors = list(cmap_custom(np.arange(len(boundaries))))
    cmap2 = mpl.colors.ListedColormap(colors, "k")
    cmap2.set_over(colors[-1])  # last color for values beyond vmax
    return cmap2

#version control
version = '004'

# Load the file
ds = xr.open_dataset(f"/home/waynedj/Projects/seaiceretention/trajectories/distances/distance_data_Parcel_OSISAF_62500m_06hr_v{version}.nc")

for t in range(0,90,5):
# data variables at time t

    cum_distances          = ds["cum_distance"].isel(time=t).values/1000
    ref_displacement       = ds["ref_displacement"].isel(time=t).values/1000
    meandering_coefficient = ds["meandering_coefficient"].isel(time=t).values

    # Reference lon/lat at time t0
    t0            = 0
    ref_longitude = ds["x"].isel(time=t0).values
    ref_latitude  = ds["y"].isel(time=t0).values

    # Create 2D lon/lat grid from reference locations
    lon_2D, lat_2D = np.meshgrid(np.unique(ref_longitude), np.unique(ref_latitude))

    # Create 3-panel plot
    fig, axes = plt.subplots(
        1, 3, figsize=(21, 7),
        subplot_kw={"projection": ccrs.SouthPolarStereo()}
    )

    # Map settings shared
    extent     = [-4000000, 4000000, -4000000, 4000000]  # adjust region
    vmin, vmax = 0, 1600  # km
    fonts      = {'textbox': 12, 'colorbar': 14, 'colorbar_ticks': 10}  # Add font sizes
    projection = ccrs.Stereographic(central_latitude=-90,true_scale_latitude=-71,central_longitude=0)
    for ax in axes:
        ax.set_extent(extent, crs=projection)
        ax.coastlines()
        ax.gridlines()
        ax.text(0.5, 0.5, f"t = {t+1} days", transform=ax.transAxes,ha='center', va='center',bbox=dict(facecolor='white', alpha=0.7, edgecolor='k'),fontsize=fonts['textbox'])

    # --- Panel 1: Cumulative distance ---
    pcm1 = axes[0].pcolormesh(lon_2D, lat_2D, cum_distances.reshape(lon_2D.shape),transform=projection, cmap=CustomColorMap1(), vmin=vmin, vmax=vmax)
    cbar1 = fig.colorbar(pcm1, ax=axes[0], orientation="horizontal", pad=0.03, shrink=0.7)
    cbar1.set_label(f"Cumulative Distance (km)", fontsize=fonts['colorbar'])
    cbar1.ax.tick_params(labelsize=fonts['colorbar_ticks'])

    # --- Panel 2: Reference displacement ---
    pcm2 = axes[1].pcolormesh(lon_2D, lat_2D, ref_displacement.reshape(lon_2D.shape),transform=projection, cmap=CustomColorMap1(), vmin=vmin, vmax=vmax)
    cbar2 = fig.colorbar(pcm2, ax=axes[1], orientation="horizontal", pad=0.03, shrink=0.7)
    cbar2.set_label(f"Reference Displacement (km)", fontsize=fonts['colorbar'])
    cbar2.ax.tick_params(labelsize=fonts['colorbar_ticks'])

    # --- Panel 3: Meandering coefficient ---
    pcm3 = axes[2].pcolormesh(lon_2D, lat_2D, meandering_coefficient.reshape(lon_2D.shape),transform=projection, cmap=CustomColorMap2(), vmin=1, vmax=8)
    cbar3 = fig.colorbar(pcm3, ax=axes[2], orientation="horizontal", pad=0.03, shrink=0.7)
    cbar3.set_label(f"Meandering Coefficient (distance/displacement)", fontsize=fonts['colorbar'])
    cbar3.ax.tick_params(labelsize=fonts['colorbar_ticks'])

    plt.savefig(f"/home/waynedj/Projects/seaiceretention/trajectories/distances/distance_data_Parcel_OSISAF_62500m_06hr_v{version}/fig_{version}_t{t+1:03d}days.png", dpi=500, bbox_inches='tight')
    plt.close()

    # %%
