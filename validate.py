import torch
import os
import numpy as np
import xarray as xr
import eccodes as ec
import argparse

from mmengine.config import Config
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

surface_shortnames = ['10v', '10u', '100v', '100u', '2t', 'tcc', 'sp', 'msl']
pressure_shortnames = ['z', 'q', 'u', 'v', 't']
pressure_levels = [1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.]

parser = argparse.ArgumentParser(description='Interp station')
parser.add_argument('--initial_time', type=str, help='initial timestamp')
args = parser.parse_args()

data_root = '/home/admin/Workspace/ec_initial_fields'
eval_root = './data/eval'
output_root = './data/output'
initial_time = args.initial_time


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

                        if shortname == 'tp':
                            data = data * 1000

                        data_dict[shortname][level] = data

                        if latitudes is None and longitudes is None:
                            latitudes = ec.codes_get_array(gid, 'latitudes').reshape((nj, ni))
                            longitudes = ec.codes_get_array(gid, 'longitudes').reshape((nj, ni))

            except ec.CodesInternalError:
                pass
            finally:
                ec.codes_release(gid)

    return data_dict, latitudes, longitudes

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

def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def latitude_weighting(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

def class_latitude_weighting(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

def rmse(predict, target):
    num_lat = predict.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=predict.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = latitude_weighting(lat_t, num_lat, s).reshape((1, 1, -1, 1))
    
    result = (predict - target) ** 2
    return (result).mean(dim=(-1, -2)).sqrt()

def eval_predict(metric, output, target):
    return metric(output, target)

def preprocess_file(filename, tp_filename):
    file_path = os.path.join(data_root, filename)
    surface_data, latitudes, longitudes = read_grib_data(file_path, surface_shortnames)
    pressure_data, _, _ = read_grib_data(file_path, pressure_shortnames, pressure_levels)

    tp_file = os.path.join(data_root, tp_filename)
    tp_data, _, _ = read_grib_data(tp_file, ['tp'])
    surface_data.update(tp_data)

    surface_data = {key_mapping[key]: value for key, value in surface_data.items()}
    output_path = os.path.join(eval_root, 'tmp.grib')
    write_netcdf_data(output_path, surface_data, pressure_data, latitudes, longitudes)
    print(f"Data saved to {output_path}")
    return

def read_ec_predict(cfg, init_time, timestamp):
    predict_field=[]
    try:
        data = xr.open_dataset(f'./data/input/era5/{timestamp}.grib')
    except Exception as e:
        print("An error occurred:", e)
        raise SystemExit("Program terminated due to an error.")
        
    idx = 0
    for vname in cfg.vnames.get('pressure'):
        vname_data = data[vname]
        for height in cfg.pressure_level:
            vdata = vname_data.sel(isobaricInhPa=height).data
            vdata = check_input(vdata)
            predict_field.append(vdata[None,:,:])
            idx += 1
            
    for vname in cfg.vnames.get('single'):
        vdata = data[vname].data
        vdata = check_input(vdata)
        predict_field.append(vdata[None,:,:])
        idx += 1
        
    predict_field = np.concatenate(predict_field, axis=0)
    
    return predict_field

def read_predict_field(cfg, init_time, timestamp):
    predict_field=[]
    var2index = {}
    index2var = {}
    try:
        pressure = xr.open_dataset(f'./data/output/{init_time}/{timestamp}_pressure.nc')
        surface = xr.open_dataset(f'./data/output/{init_time}/{timestamp}_surface.nc')
    except Exception as e:
        print("An error occurred:", e)
        raise SystemExit("Program terminated due to an error.")
        
    idx = 0
    for vname in cfg.vnames.get('pressure'):
        vname_data = pressure[vname]
        for height in cfg.pressure_level:
            vdata = vname_data.sel(isobaricInhPa=height).data
            vdata = check_input(vdata[0])
            iheight = int(height)
            var2index[f"{vname}{iheight}"] = idx
            index2var[idx] = f"{vname}{iheight}"
            predict_field.append(vdata[None,:,:])
            idx += 1
            
    for vname in cfg.vnames.get('single'):
        vdata = surface[vname].data
        vdata = check_input(vdata[0])
        var2index[f"{vname}"] = idx
        index2var[idx] = f"{vname}"
        predict_field.append(vdata[None,:,:])
        idx += 1
        
    predict_field = np.concatenate(predict_field, axis=0)
    
    return predict_field, var2index, index2var

def check_input(vdata):
    input_shape = (2001, 4000) 
    new_array = zoom(vdata, 
                     (input_shape[0] / vdata.shape[0],
                      input_shape[1] / vdata.shape[1]),
                     order=1)  #
    return new_array

def read_initial_field(cfg):
    input_initial_field=[]
    var2index = {}
    index2var = {}
    try:
        data = xr.open_dataset(f'./data/eval/tmp.grib')
    except Exception as e:
        print("An error occurred:", e)
        raise SystemExit("Program terminated due to an error.")

    idx = 0
    for vname in cfg.vnames.get('pressure'):
        vname_data = data[vname]
        for height in cfg.pressure_level:
            vdata = vname_data.sel(isobaricInhPa=height).data
            vdata = check_input(vdata)
            vdata = np.roll(vdata, 2000, axis=1)
            iheight = int(height)
            var2index[f"{vname}{iheight}"] = idx
            index2var[idx] = f"{vname}{iheight}"
            input_initial_field.append(vdata[None,:,:])
            idx += 1

    for vname in cfg.vnames.get('single'):
        vdata = data[vname].data
        vdata = check_input(vdata)
        vdata = np.roll(vdata, 2000, axis=1)
        var2index[f"{vname}"] = idx
        index2var[idx] = f"{vname}"
        input_initial_field.append(vdata[None,:,:])
        idx += 1

    input_initial_field = np.concatenate(input_initial_field, axis=0)
    return input_initial_field, var2index, index2var


time_obj = datetime.strptime(initial_time, '%Y-%m-%dT%H:%M:%S')
cfg = Config.fromfile('./config/fengwu_ghr_cfg.py')

for i in range(40):
    out_time_obj = time_obj + timedelta(hours=6*(i+1))
    out_time = out_time_obj.strftime('%Y-%m-%dT%H:%M:%S')
    ec_out_time = out_time_obj.strftime('%m%d%H')

    predict, var2index, index2var = read_predict_field(cfg, initial_time, out_time)
    ec_pred = read_ec_predict(cfg, initial_time, ec_out_time)

    time_str = out_time_obj.strftime('%m%d%H')
    time_hour = out_time_obj.strftime('%H')
    tp_time = out_time_obj - timedelta(hours=6)
    tp_str = tp_time.strftime('%m%d%H')

    print(out_time, ec_out_time, time_str)
    if time_hour in ['06', '18']:
        initial_file = f'A1S{time_str}00{time_str}011'
        tp_file = f'A1D{tp_str}00{time_str}001'
    elif time_hour in ['00', '12']:
        initial_file = f'A1D{time_str}00{time_str}011'
        tp_file = f'A1S{tp_str}00{time_str}001'
    else:
        raise ValueError('Not a valid time')
        
    preprocess_file(initial_file, tp_file)
    target, var2index1, index2var1 = read_initial_field(cfg)

    predict = torch.from_numpy(predict).unsqueeze(0)
    ec_pred = torch.from_numpy(ec_pred).unsqueeze(0)
    target = torch.from_numpy(target).unsqueeze(0)
    error = eval_predict(rmse, predict, target)[0]
    ec_error = eval_predict(rmse, ec_pred, target)[0]

    print('step: ', i, error[var2index['t2m']])
    print('step: ', i, error[var2index['msl']])
    print('step: ', i, error[var2index['sp']])
    print('step: ', i, error[var2index['z500']])
    #print('step: ', i, ec_error[var2index['t2m']])
