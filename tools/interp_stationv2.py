import os
import numpy as np
import xarray as xr
import pandas as pd
from metpy.calc import wind_speed, wind_direction
from metpy.units import units
from tqdm import trange

save_root = './data/shanxi'
initial_time = '2023-06-01T00:00:00'

os.path.exists("result") or os.mkdir("result")
ds = xr.open_dataset(f"{save_root}/FengWu_GHR_Shanxi_{initial_time}.nc")

df = pd.read_excel(f"./tools/station_of_shanxi.xls", dtype=str)
df = df[["ID",  "PLANT_TYPE", "LONGITUDE", "LATITUDE"]]
lon, lat = xr.DataArray(df.iloc[:, -2:].values.astype(float), dims=(("points", "lonlat"))).T
np_points = len(lon)

ds = ds.interp(longitude=lon, latitude=lat)
#100hpa插值输出后注释

aim_lev = ds.isobaricInhPa.astype(int).data
# aim_lev = np.sort(np.append(plev, 100))[::-1]
# ds = ds.interp(isobaricInhPa=aim_lev)

wd = wind_direction(ds.u.data * units("m/s"), ds.v.data * units("m/s")).__array__() + ds.u * 0
ws = wind_speed(ds.u.data * units("m/s"), ds.v.data * units("m/s")).__array__() + ds.u * 0
ds = ds.assign(wd=wd, ws=ws)
wd100 = wind_direction(ds.u100.data * units("m/s"), ds.v100.data * units("m/s")).__array__() + ds.u100 * 0
ws100 = wind_speed(ds.u100.data * units("m/s"), ds.v100.data * units("m/s")).__array__() + ds.u100 * 0
wd10 = wind_direction(ds.u10.data * units("m/s"), ds.v10.data * units("m/s")).__array__() + ds.u10 * 0
ws10 = wind_speed(ds.u10.data * units("m/s"), ds.v10.data * units("m/s")).__array__() + ds.u10 * 0
ds = ds.assign(wd100=wd100, wd10=wd10, ws100=ws100, ws10=ws10)


time = pd.date_range(start=ds.time.data[0], periods=4 * 24 * 10 + 1, freq='15min')
nt = len(time)
ds = ds.interp(time=time, kwargs={"fill_value": "extrapolate"})

#ssr注释
# ssr = xr.open_dataset("Surface_solar_radiation_downwards.nc")["var169"]
# ssr = ssr.interp(lon=lon, lat=lat).data
# ssr = ssr * np.random.uniform(0.7, 1.3, (nt, 1, 1))
ssr = np.random.uniform(0.7, 1.3, (nt, 1, np_points))

vars_ = "msl ws100 wd100 ws10 wd10 t2m tp6h tcc".split()
data = np.stack([ds[v].data for v in "z ws wd q t".split()], 1).reshape(nt, -1, np_points)
data = np.hstack([data, np.stack([ds[v].data for v in vars_], 1), ssr])

columns = [f"{v}_{lev}" for v in "z ws wd q t".split() for lev in aim_lev] + "msl ws100 wd100 ws10 wd10 t2m tp tcc ssr".split()


for i in trange(np_points):
    id_, plant_type = df.iloc[i, :2].values
    dat = data[..., i]
    daf = pd.DataFrame(dat, index=time, columns=columns)
    daf.index.name = "time"
    os.makedirs(f"{save_root}/station/{initial_time}", exist_ok=True)
    daf.to_csv(f"{save_root}/station/{initial_time}/{id_}_{plant_type}_{initial_time}.csv") 

