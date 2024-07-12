import os
import numpy as np
import xarray as xr
import eccodes as ec
import argparse
import pandas as pd
import subprocess

from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy.ndimage import zoom


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
    # import pdb
    # pdb.set_trace()
    longitudes = np.linspace(0, 359.9, longitudes.shape[1])
    longitudes = np.tile(longitudes, [longitudes.shape[0], 1])
    pressure_data, _, _ = read_grib_data(file_path, pressure_shortnames, pressure_levels)
    # pdb.set_trace()

    tp_file = os.path.join(data_root, tp_filename)
    tp_data, _, _ = read_grib_data(tp_file, ['tp'])
    # pdb.set_trace()
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


surface_shortnames = ['10v', '10u', '100v', '100u', '2t', 'tcc', 'sp', 'msl']
pressure_shortnames = ['z', 'q', 'u', 'v', 't']
pressure_levels = [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.]

parser = argparse.ArgumentParser(description='Interp station')
parser.add_argument('--initial_time', type=str, help='initial timestamp')
args = parser.parse_args()

ghr_root = '/home/admin/NWP/FengWu_GHR/data/input/era5'
data_root = '/home/sftpuser/ec_initial_fields'
save_root = './data/ec_pred'
cur_dir = os.getcwd()
initial_times = args.initial_time.split(' ')

start_time = pd.Timestamp(initial_times[0])
end_time = pd.Timestamp(initial_times[1])
time_range = pd.date_range(start=start_time, end=end_time, freq='6H').strftime('%Y-%m-%dT%H:%M:%S')

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
        for i in range(1, 28):
            pred_time = time_obj + timedelta(hours=6*i)
            pred_str = pred_time.strftime('%m%d%H')
            pred_file = f'A1D{time_str}00{pred_str}001'
            print(pred_file)

            pred_save_str = pred_time.strftime('%Y-%m-%dT%H:%M:%S')
            save_str = os.path.join(initial_time, pred_save_str)
            os.makedirs(os.path.join(save_root, initial_time), exist_ok=True)

            os.chdir(data_root)
            command = f"bzip2 -d {pred_file}.bz2"
            process = subprocess.Popen(command, shell=True)
            process.wait()
            os.chdir(cur_dir)

            try:
                preprocess_file(pred_file, pred_file, save_str)
            except:
                continue
    else:
        raise ValueError('Not a valid time')

    # print(initial_file)
    # preprocess_file(initial_file, tp_file, initial_time)
    # preprocess_file(pred_file, pred_file, 'pred' + initial_time)