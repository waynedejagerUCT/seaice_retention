#%%
import xarray as xr
import numpy as np
import numpy as np
import pandas as pd
import dispersion_utils as utils

def compute_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points on a 
    polar stereographic projection (Cartesian plane, meter units).

    Parameters
    ----------
    x1, y1 : float or array-like
        Coordinates of the first point (meters).
    x2, y2 : float or array-like
        Coordinates of the second point (meters).

    Returns
    -------
    distance : float or array-like
        Distance in meters.
    """
    dx = np.subtract(x2, x1)
    dy = np.subtract(y2, y1)
    return np.sqrt(np.add(np.power(dx, 2), np.power(dy, 2)))

import numpy as np
import xarray as xr

#version control
version = '004'

# fetch trajectory data
ds1     = xr.open_zarr(f"/home/waynedj/Projects/seaiceretention/trajectories/Parcel_OSISAF_62500m_06hr_v{version}.zarr", decode_timedelta=True)
p_total = len(ds1.trajectory)
t_days  = len(ds1.obs)

# arrays for outputs
step_distance      = np.empty((p_total, t_days))
cum_distance       = np.empty((p_total, t_days))
ref_displacement   = np.empty((p_total, t_days))

# initialize start values
step_distance[:, 0]    = 0.0
cum_distance[:, 0]     = 0.0
ref_displacement[:, 0] = 0.0

# lon/lat storage (optional, for reference in output)
x       = np.empty((p_total, t_days))
y       = np.empty((p_total, t_days))
x[:, 0] = ds1['lon'][:, 0].values
y[:, 0] = ds1['lat'][:, 0].values

# loop through timesteps
for time_step in range(1, t_days):
    # stepwise displacement (t vs t-1)
    step_distance[:, time_step] = compute_distance(ds1['lon'][:, time_step].values,
                                                   ds1['lat'][:, time_step].values,
                                                   ds1['lon'][:, time_step - 1].values,
                                                   ds1['lat'][:, time_step - 1].values)

    # cumulative displacement
    cum_distance[:, time_step] = cum_distance[:, time_step - 1] + step_distance[:, time_step]

    # reference displacement (t vs 0)
    ref_displacement[:, time_step] = compute_distance(ds1['lon'][:, time_step].values,
                                                      ds1['lat'][:, time_step].values,
                                                      ds1['lon'][:, 0].values,
                                                      ds1['lat'][:, 0].values)

    # store lon/lat values
    x[:, time_step] = ds1['lon'][:, time_step].values
    y[:, time_step] = ds1['lat'][:, time_step].values

# Compute meandering coefficient: M(t) = cum_distance / ref_displacement
eps = 1e-8  # avoid divide by zero
meandering_coefficient = cum_distance / np.where(ref_displacement <= eps, np.nan, ref_displacement)

# build xarray Dataset
ds_output = xr.Dataset(
    {
        "step_distance": (["particle", "time"], step_distance),
        "cum_distance" : (["particle", "time"], cum_distance),
        "ref_displacement": (["particle", "time"], ref_displacement),
        "meandering_coefficient": (["particle", "time"], meandering_coefficient),
        "x": (["particle", "time"], x),
        "y": (["particle", "time"], y),
    },
    coords={
        "particle": ds1.trajectory.values,
        "time": ds1.obs.values,
    }
)

# add attrs for clarity
ds_output["step_distance"].attrs["description"] = "Displacement between consecutive timesteps"
ds_output["cum_distance"].attrs["description"] = "Cumulative distance travelled up to each timestep"
ds_output["ref_displacement"].attrs["description"] = "Displacement from reference (time=0) location"
ds_output["meandering_coefficient"].attrs["description"] = "Meandering coefficient (M(t) = cum_distance / ref_displacement)"
ds_output["step_distance"].attrs["units"] = "km"
ds_output["cum_distance"].attrs["units"] = "km"
ds_output["ref_displacement"].attrs["units"] = "km"
ds_output["meandering_coefficient"].attrs["units"] = "dimensionless"
ds_output["x"].attrs["units"] = "m"
ds_output["y"].attrs["units"] = "m"

# Save the dataset
ds_output.to_netcdf(f"/home/waynedj/Projects/seaiceretention/trajectories/distances/distance_data_Parcel_OSISAF_62500m_06hr_v{version}.nc")


# %%
