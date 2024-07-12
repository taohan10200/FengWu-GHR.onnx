import os
import numpy as np
import xarray as xr
import eccodes as ec
import argparse
import glob
import pandas as pd
import subprocess

from metpy.calc import wind_speed, wind_direction
from metpy.units import units
from datetime import datetime, timedelta
from netCDF4 import Dataset


key_mapping = {
    '10v': 'v10',
    '10u': 'u10',
    '100v': 'v100',
    '100u': 'u100',
    '2t': 't2m',
    'tcc': 'tcc',
    'sp': 'sp',
    'msl': 'msl',
    'tp': 'tp6h'
}

def read_grib_data(filename, shortnames, levels=None):
    data_dict = {shortname: {} for shortname in shortnames}
    latitudes = None
    longitudes = None
    
    with open(filename, 'rb') as grib_file:
        while True:
            gid = ec.codes_grib_new_from_file(grib_file)
            if gid is None:
                break

            try:
                shortname = ec.codes_get(gid, 'shortName')
                level = ec.codes_get(gid, 'level')
                
                if shortname in shortnames:
                    if (levels is None and level == 0) or (levels is not None and level in levels):
                        ni = ec.codes_get(gid, 'Ni')
                        nj = ec.codes_get(gid, 'Nj')
                        data = ec.codes_get_values(gid).reshape((nj, ni))
                        data = np.roll(data, ni//2, axis=1)

                        data_dict[shortname][level] = data

                        if latitudes is None and longitudes is None:
                            latitudes = ec.codes_get_array(gid, 'latitudes').reshape((nj, ni))
                            longitudes = ec.codes_get_array(gid, 'longitudes').reshape((nj, ni))

            except ec.CodesInternalError:
                pass
            finally:
                ec.codes_release(gid)

    return data_dict, latitudes, longitudes

def preprocess_file(filename, tp_filename, output_name):
    file_path = os.path.join(data_root, filename)
    surface_data, latitudes, longitudes = read_grib_data(file_path, surface_shortnames)
    longitudes = np.linspace(0, 359.9, longitudes.shape[1])
    longitudes = np.tile(longitudes, [longitudes.shape[0], 1])
    pressure_data, _, _ = read_grib_data(file_path, pressure_shortnames, pressure_levels)

    tp_file = os.path.join(data_root, tp_filename)
    tp_data, _, _ = read_grib_data(tp_file, ['tp'])
    surface_data.update(tp_data)

    surface_data = {key_mapping[key]: value for key, value in surface_data.items()}
    output_path = os.path.join(save_root, f'{output_name}.grib')
    write_netcdf_data(output_path, surface_data, pressure_data, latitudes, longitudes)
    print(f"Data saved to {output_path}")
    return

def write_netcdf_data(filename, surface_data, pressure_data, latitudes, longitudes):
    with Dataset(filename, 'w', format='NETCDF4') as ncfile:
        lat_dim = ncfile.createDimension('latitude', latitudes.shape[0])
        lon_dim = ncfile.createDimension('longitude', longitudes.shape[1])
        level_dim = ncfile.createDimension('isobaricInhPa', len(pressure_data[next(iter(pressure_data))].keys()))

        latitudes_var = ncfile.createVariable('latitude', np.float32, ('latitude',))
        longitudes_var = ncfile.createVariable('longitude', np.float32, ('longitude',))
        level_var = ncfile.createVariable('isobaricInhPa', np.int32, ('isobaricInhPa',))

        latitudes_var[:] = latitudes[:, 0]
        longitudes_var[:] = longitudes[0, :]
        level_var[:] = pressure_levels

        for shortname, data in surface_data.items():
            var = ncfile.createVariable(shortname, np.float32, ('latitude', 'longitude'))
            var[:, :] = data[0]

        for shortname, levels_data in pressure_data.items():
            var = ncfile.createVariable(shortname, np.float32, ('isobaricInhPa', 'latitude', 'longitude'))
            for i, (level, data) in enumerate(levels_data.items()):
                idx = pressure_levels.index(level)
                var[idx, :, :] = data

def ghr_inference(filename):
    timestamp = os.path.splitext(filename)[0]
    command = f"python -u fengwu_ghr_inference_torch_deploy.py --timestamp={timestamp} --config='config/fengwu_ghr_cfg_torch_deploy.py' --gpu=0"

    process = subprocess.Popen(command, shell=True)
    process.wait()

def classify_tp6h(value):
    if value < 0.1:
        return '零星小雨'
    elif 0.1 <= value <= 9.9:
        return '小雨'
    elif 10.0 <= value <= 24.9:
        return '中雨'
    elif 25.0 <= value <= 49.9:
        return '大雨'
    elif 50.0 <= value <= 99.9:
        return '暴雨'
    elif 100.0 <= value <= 249.9:
        return '大暴雨'
    else:
        return '特大暴雨'

def classify_wind_speed(speed):
    if speed < 0.3:
        return '0 无风'
    elif 0.3 <= speed < 1.6:
        return '1 软风'
    elif 1.6 <= speed < 3.4:
        return '2 轻风'
    elif 3.4 <= speed < 5.5:
        return '3 微风'
    elif 5.5 <= speed < 8.0:
        return '4 和风'
    elif 8.0 <= speed < 10.8:
        return '5 清劲风'
    elif 10.8 <= speed < 13.9:
        return '6 强风'
    elif 13.9 <= speed < 17.2:
        return '7 疾风'
    elif 17.2 <= speed < 20.8:
        return '8 大风'
    elif 20.8 <= speed < 24.5:
        return '9 烈风'
    elif 24.5 <= speed < 28.5:
        return '10 狂风'
    elif 28.5 <= speed < 32.6:
        return '11 暴风'
    elif 32.6 <= speed < 37.0:
        return '12 台风，或飓风'
    elif 37.0 <= speed < 41.5:
        return '13 台风'
    elif 41.5 <= speed < 46.2:
        return '14 强台风'
    elif 46.2 <= speed < 51.0:
        return '15 强台风'
    elif 51.0 <= speed < 56.1:
        return '16 超强台风'
    else:
        return '17 超强台风'
    
def get_result(initial_time, lon, lat, start_time):
    file_lists = glob.glob(
        os.path.join(f'{output_root}/{initial_time}', "*surface.nc")
    )
    file_lists.sort()

    time_list, wd_list, ws_list, tp6h_list, t2m_list = [], [], [], [], []
    for idx, file_name in enumerate(file_lists):
        nc_data = xr.open_dataset(file_name)

        u10_pred = nc_data['u10'].interp(longitude=lon, latitude=lat).values[0] * units('m/s')
        v10_pred = nc_data['v10'].interp(longitude=lon, latitude=lat).values[0] * units('m/s')
        wd_pred = wind_direction(u10_pred, v10_pred).__array__()
        ws_pred = wind_speed(u10_pred, v10_pred).__array__()
        
        t2m_pred = nc_data['t2m'].interp(longitude=lon, latitude=lat).values[0] - 273.15
        tp6h_pred = nc_data['tp6h'].interp(longitude=lon, latitude=lat).values[0]

        timestamp = os.path.basename(file_name).strip('_surface.nc')
        time_list.append(timestamp)
        wd_list.append(wd_pred)
        ws_list.append(ws_pred)
        tp6h_list.append(tp6h_pred)
        t2m_list.append(t2m_pred)

    offset = time_list.index(start_time)

    # time
    timestamp = []
    for i in range(offset, len(time_list), 4):
        if i + 3 < len(time_list):
            timestamp.append(time_list[i])

    # tp
    tp = []
    for i in range(offset, len(tp6h_list), 4):
        if i + 3 < len(tp6h_list):
            tp.append((tp6h_list[i] + tp6h_list[i+1] + tp6h_list[i+2] + tp6h_list[i+3]))

    # ws
    ws = []
    for i in range(offset, len(ws_list), 4):
        if i + 3 < len(ws_list):
            ws.append((ws_list[i] + ws_list[i+1] + ws_list[i+2] + ws_list[i+3]) / 4)

    # t2m
    hour2, hour14 = [], []
    for i in range(offset, len(t2m_list), 4):
        if i + 2 < len(t2m_list):
            hour2.append(t2m_list[i])
            hour14.append(t2m_list[i + 2])

    df = pd.DataFrame({
        'time': timestamp,
        'tp6h': tp,
        'ws': ws,
        'hour2': hour2,
        'hour14': hour14,
    })
    df = df.round(1)
    df['降水等级'] = df['tp6h'].apply(classify_tp6h)
    df['风力等级'] = df['ws'].apply(classify_wind_speed)

    df.to_excel(f'{initial_time}.xlsx', index=False)

surface_shortnames = ['10v', '10u', '100v', '100u', '2t', 'tcc', 'sp', 'msl']
pressure_shortnames = ['z', 'q', 'u', 'v', 't']
pressure_levels = [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.]

parser = argparse.ArgumentParser(description='Interp station')
parser.add_argument('--initial_time', type=str, help='initial timestamp')
parser.add_argument('--get_report', action='store_true', help='initial timestamp')
parser.add_argument('--start_time', type=str, help='initial timestamp')
args = parser.parse_args()

data_root = '/home/admin/Workspace/ec_initial_fields'
save_root = '/mnt/prediction/NWP/FengWu_GHR/data/input/era5'
output_root = '/mnt/prediction/NWP/FengWu_GHR/data/output'
initial_times = args.initial_time.split(' ')
lon = 121.48389
lat = 31.18672

start_time = pd.Timestamp(initial_times[0])
end_time = pd.Timestamp(initial_times[1])
time_range = pd.date_range(start=start_time, end=end_time, freq='6h').strftime('%Y-%m-%dT%H:%M:%S')

for initial_time in time_range:
    time_obj = datetime.strptime(initial_time, '%Y-%m-%dT%H:%M:%S')
    time_str = time_obj.strftime('%m%d%H')
    time_hour = time_obj.strftime('%H')
    tp_time = time_obj - timedelta(hours=6)
    tp_str = tp_time.strftime('%m%d%H')

    if time_hour in ['06', '18']:
        initial_file = f'A1S{time_str}00{time_str}011'
        tp_file = f'A1D{tp_str}00{time_str}001'
    elif time_hour in ['00', '12']:
        initial_file = f'A1D{time_str}00{time_str}011'
        tp_file = f'A1S{tp_str}00{time_str}001'
    else:
        raise ValueError('Not a valid time')

    preprocess_file(initial_file, tp_file, initial_time)
    if args.get_report:
        ghr_inference(initial_time)
        get_result(initial_time, lon, lat, args.start_time)
