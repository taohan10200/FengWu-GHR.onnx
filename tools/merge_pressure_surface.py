import os
import numpy as np
import xarray as xr
import pandas as pd

save_root = './data/shanxi'
initial_time = '2023-06-01T00:00:00'
input_root = f'./data/output/{initial_time}'
os.makedirs(save_root, exist_ok=True)


ds1 = xr.open_mfdataset(f"{input_root}/*pressure.nc")
ds2 = xr.open_mfdataset(f"{input_root}/*surface.nc")
ds = xr.merge([ds1, ds2]).sortby("latitude").sel(longitude=slice(107.5, 117.0), latitude=slice(32.0, 42.85)) #longitude=slice(110, 114.6), latitude=slice(34.5, 40.85)

plev = ds.isobaricInhPa.astype(int).data
aim_lev = np.sort(np.append(plev, 100))[::-1]
ds = ds.interp(isobaricInhPa=aim_lev)

time = pd.date_range(ds.time.data[0], ds.time.data[-1], freq='15min')
ds = ds.interp(time=time).compute()
ds.to_netcdf(f"{save_root}/FengWu_GHR_Shanxi_{initial_time}.nc")

print(xr.open_dataset(f"{save_root}/FengWu_GHR_Shanxi_{initial_time}.nc"))
