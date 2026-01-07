
#%%
import xarray as xr
import numpy as np
from parcels import Field, FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4
from parcels.tools.statuscodes import StatusCode
from datetime import timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Open NEMO drift file
fname = 'ORCA2_6h_20190101_20191231_icemod.nc'
fdir  = '/home/waynedj/Data/seaicedrift/NEMO/'
ds    = xr.open_dataset(fdir + fname)

# Extract coords (latlon) and velocity (m/s)
longitude    = ds["nav_lon_grid_T"].values
latitude     = ds["nav_lat_grid_T"].values
u            = np.nan_to_num(ds["sivelu"].values, nan=0.0)
v            = np.nan_to_num(ds["sivelv"].values, nan=0.0)
u_0          = np.zeros_like(u)
v_0          = np.zeros_like(v)
time         = ds["time_centered"].values

# Convert time to seconds since start of the dataset for Parcels
time_seconds = (time - time[0]).astype('timedelta64[s]').astype(np.float64)

# Create Fields
Ufield       = Field(name="U", data=u, lon=longitude, lat=latitude, time=time_seconds, mesh="spherical")
Vfield       = Field(name="V", data=v, lon=longitude, lat=latitude, time=time_seconds, mesh="spherical")
Ufield_0     = Field(name="U", data=u_0, lon=longitude, lat=latitude, time=time_seconds, mesh="spherical")
Vfield_0     = Field(name="V", data=v_0, lon=longitude, lat=latitude, time=time_seconds, mesh="spherical")

# Create a FieldSet
fieldset     = FieldSet(Ufield, Vfield)
fieldset_0   = FieldSet(Ufield_0, Vfield_0)


# Calculate start time in seconds (for index 604, which is 03h00 1 June 2019)
start_time_index   = 604
start_time_seconds = time_seconds[start_time_index]

# define drop locations
drop_lon                 = np.arange(-178, 178, 0.5)
drop_lat                 = np.arange( -78, -50, 0.5)
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

output_file = pset.ParticleFile(name="/home/waynedj/Projects/seaiceretention/trajectories/Parcel_NEMO_1deg_06hr_ds1_005.zarr", outputdt=timedelta(days=1))

pset.execute([AdvectionRK4, KillIfOutOfBounds],
             runtime=timedelta(days=10),
             dt=timedelta(hours=12),
             output_file=output_file,)

# ds2: create particle set and execute
pset = ParticleSet.from_list(fieldset=fieldset_0, 
                            pclass=JITParticle, 
                            lon=drop_lon_1D,
                            lat=drop_lat_1D,
                            time=[start_time_seconds])

output_file = pset.ParticleFile(name="/home/waynedj/Projects/seaiceretention/trajectories/Parcel_NEMO_1deg_06hr_ds2_005.zarr", outputdt=timedelta(days=1))

pset.execute([AdvectionRK4, KillIfOutOfBounds],
             runtime=timedelta(days=10),
             dt=timedelta(hours=12),
             output_file=output_file,)














# %%
