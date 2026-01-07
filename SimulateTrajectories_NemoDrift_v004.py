
#%%
import xarray as xr
import numpy as np
from parcels import Field, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4
from parcels.tools.statuscodes import StatusCode
from datetime import timedelta

version = '005'

def make_curvilinear_periodic(longitude, latitude, u, v):
    """
    Stitch a curvilinear NEMO grid to remove longitudinal discontinuities for Parcels.
    Automatically detects the largest jump, rolls the grid so the jump is at the edge,
    and duplicates first/last columns to make the grid continuous.

    Parameters
    ----------
    longitude : np.ndarray [ny, nx]
        2D longitude coordinates
    latitude : np.ndarray [ny, nx]
        2D latitude coordinates
    u : np.ndarray [time, ny, nx]
        Zonal velocity
    v : np.ndarray [time, ny, nx]
        Meridional velocity

    Returns
    -------
    longitude_new : np.ndarray [ny, nx+2]
    latitude_new : np.ndarray [ny, nx+2]
    u_new : np.ndarray [time, ny, nx+2]
    v_new : np.ndarray [time, ny, nx+2]
    """

    # --- Step 1: Normalize longitudes to 0–360 ---
    longitude = longitude % 360

    # --- Step 2: Detect largest jump along x ---
    dlon_x = np.diff(longitude, axis=1)
    max_jump_idx = np.unravel_index(np.argmax(np.abs(dlon_x)), dlon_x.shape)
    shift = longitude.shape[1] - max_jump_idx[1] - 1

    # --- Step 3: Roll grid so largest jump is at the edge ---
    longitude_rolled = np.roll(longitude, shift, axis=1)
    latitude_rolled  = np.roll(latitude, shift, axis=1)
    u_rolled         = np.roll(u, shift, axis=2)
    v_rolled         = np.roll(v, shift, axis=2)

    # --- Step 4: Duplicate first/last columns to close the seam ---
    lon_first = longitude_rolled[:, :1]
    lon_last  = longitude_rolled[:, -1:]
    lat_first = latitude_rolled[:, :1]
    lat_last  = latitude_rolled[:, -1:]

    u_first = u_rolled[:, :, :1]
    u_last  = u_rolled[:, :, -1:]
    v_first = v_rolled[:, :, :1]
    v_last  = v_rolled[:, :, -1:]

    longitude_new = np.hstack((lon_last - 360, longitude_rolled, lon_first + 360))
    latitude_new  = np.hstack((lat_last, latitude_rolled, lat_first))
    u_new         = np.concatenate((u_last, u_rolled, u_first), axis=2)
    v_new         = np.concatenate((v_last, v_rolled, v_first), axis=2)

    return longitude_new, latitude_new, u_new, v_new

# Open NEMO drift file
fname     = 'ORCA025_combined_icemod_20190601_20190731.nc'
fdir      = '/home/waynedj/Data/NEMO/ORCA_025deg/'
ds        = xr.open_dataset(fdir + fname)

# Extract coords (latlon) and velocity (m/s)
lat_index_max = 348 #approx -40 degS
lat_index_min = 40  #approx -74 degS
latitude  = ds["nav_lat"].values[lat_index_min:lat_index_max, :]
longitude = ds["nav_lon"].values[lat_index_min:lat_index_max, :]
u         = np.nan_to_num(ds["sivelu"].values[:, lat_index_min:lat_index_max, :], nan=0.0)
v         = np.nan_to_num(ds["sivelv"].values[:, lat_index_min:lat_index_max, :], nan=0.0)
time      = ds["time_centered"].values

# Roll and stitch grid to remove discontinuities
longitude, latitude, u, v = make_curvilinear_periodic(longitude, latitude, u, v)

# Convert time to seconds since start of the dataset for Parcels
time_seconds = (time - time[0]).astype('timedelta64[s]').astype(np.float64)

# Create Fields
Ufield       = Field(name="U", data=u, lon=longitude, lat=latitude, time=time_seconds, mesh="spherical")
Vfield       = Field(name="V", data=v, lon=longitude, lat=latitude, time=time_seconds, mesh="spherical")

# Create a FieldSet
fieldset     = FieldSet(Ufield, Vfield)

# Calculate start time in seconds (for index 0, which is 00h00 1 June 2019)
start_time_index   = 0
start_time_seconds = time_seconds[start_time_index]

# define drop locations
drop_lon                 = np.arange(0, 360, 0.5)
drop_lon                 = drop_lon % 360
drop_lat                 = np.arange(-78, -54, 0.5)
drop_lon_2D, drop_lat_2D = np.meshgrid(drop_lon, drop_lat)
drop_lon_1D              = drop_lon_2D.ravel()
drop_lat_1D              = drop_lat_2D.ravel()

# Handle errors inside the kernel loop
def KillIfOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

# ds1: create particle set and execute
pset = ParticleSet.from_list(fieldset=fieldset, 
                            pclass=JITParticle, 
                            lon=drop_lon_1D,
                            lat=drop_lat_1D,
                            time=[start_time_seconds])

pset.populate_indices()

output_file = pset.ParticleFile(name=f"/home/waynedj/Data/intermediate/seaice_retention/trajectories/NEMO_025deg_06hr_v{version}.zarr", outputdt=timedelta(days=1))

pset.execute([AdvectionRK4, KillIfOutOfBounds],
             runtime=timedelta(days=60),
             dt=timedelta(hours=6),
             output_file=output_file,)


# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

projection = ccrs.Stereographic(central_latitude=-90,true_scale_latitude=-71,central_longitude=0)

version = '005'

# Set up figure and axes
fig = plt.figure(figsize=(18, 18))
ax = plt.axes(projection=projection)
extent = [-3500000, 3500000, -3500000, 3500000]
#extent  = [-3200000,  500000,   500000, 4000000]
ax.set_extent(extent, crs=projection)
ax.coastlines()
ax.gridlines(draw_labels=True)

ds = xr.open_zarr(f"/home/waynedj/Data/intermediate/seaice_retention/trajectories/NEMO_025deg_06hr_v{version}.zarr", decode_timedelta=True)


start   = True
end     = False
traj    = False
density = 10

# Plot each trajectory
for traj_idx in range(0, len(ds.trajectory), density):
    if start:
        ax.plot(
            ds.lon[traj_idx, 0],
            ds.lat[traj_idx, 0],
            'go',  # Green dot for start
            transform=ccrs.PlateCarree(),
            markersize=5,
            label='Start'
        )
    if end:
        ax.plot(
            ds.lon[traj_idx, -1],
            ds.lat[traj_idx, -1],
            'rd',  # Red x for end
            transform=ccrs.PlateCarree(),
            markersize=5,
            label='End'
        )
    if traj:            
        ax.plot(
            ds.lon[traj_idx],
            ds.lat[traj_idx],
            color='k',
            linewidth=1,
            transform=ccrs.PlateCarree(),
            alpha=0.6
            )

#plt.legend(loc='upper left')






# %%
