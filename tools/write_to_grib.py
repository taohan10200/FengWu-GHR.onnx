# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import torch
import pytz
import os
import numpy as np
import xarray as xr
from typing import Optional, Tuple, Union, Dict
from pathlib import Path 
from cfgrib.xarray_to_grib import to_grib
from datetime import datetime

grib_para = {
"z": {"Name":"Geopotential",             "ShortName":"z",   'Unit':"m^2 s^-2", "ParaID": 129},
"t": {"Name":"Temperature",             "ShortName":"t",   'Unit':"K",          "ParaID": 130},
"u": {"Name":"Temperature",              "ShortName":"u",   'Unit':"m s^-1",   "ParaID": 131},
"v": {"Name":"V component of wind",      "ShortName":"v",   'Unit':"m s^-1",   "ParaID": 132},
"q": {"Name":"Specific humidity",        "ShortName":"q",   'Unit':"kg kg^-1", "ParaID": 133},
"w": {"Name":"Vertical velocity",        "ShortName":"w",   'Unit':"Pa s^-1",  "ParaID": 135},
"u10": {"Name":"10m V component of wind",        "ShortName":"u10",   'Unit':"m s^-1",  "ParaID": 135},
"v10": {"Name":"10m V component of wind",        "ShortName":"v10",   'Unit':"m s^-1",  "ParaID": 135},
"u100": {"Name":"100m V component of wind",        "ShortName":"u100",   'Unit':"m s^-1",  "ParaID": 135},
"v100": {"Name":"100m V component of wind",        "ShortName":"v100",   'Unit':"m s^-1",  "ParaID": 135},
"tp6h": {"Name":"Total 6H Precipitation", "ShortName":"tp6h",   'Unit':"mm",  "ParaID": 228},
"tp": {"Name":"Total 1H Precipitation", "ShortName":"tp",   'Unit':"mm",  "ParaID": 228},
"t2m": {"Name":"2 meter temperature", "ShortName":"t2m",   'Unit':"K",  "ParaID": 167},
"msl": {"Name":"mean sea level pressure", "ShortName":"msl",   'Unit':"Pa",  "ParaID": 129.151},
"tcc": {"Name":"Total cloud cover", "ShortName":"tcc",   'Unit':"0-1",  "ParaID": 164},
"sp":{"Name":"Surface pressure",          "ShortName":"sp",   'Unit':"Pa",        "ParaID": 134},
"ssr": {"Name":"Surface net short-wave (solar) radiation", "ShortName":"ssr",   'Unit':"J m-2",  "ParaID": 176},
"ssr6h": {"Name":"Surface net short-wave (solar) radiation (6 hour)", "ShortName":"ssr6h",   'Unit':"J m-2",  "ParaID": 176},
}
def write_grib(data_sample: Union[torch.Tensor, np.ndarray], 
               save_root: Union[str,Path]=None, 
               channels_to_vname: Dict=None, 
               filter_dict: list=['z_500','z_850', 't_500', 't_850','tp6h', 'u10', 'v10'],
               s3_client: object=None,
               merge_pressure_surface = True
               )->None:
    """
    args: 
    datasample={
                'pred_label':{
                                '2022-01-01T06:00:00':[69, 721, 1440],
                                '2022-01-01T12:00:00':[69, 721, 1440],
                                '2022-01-01T18:00:00':[69, 721, 1440],
                                },
                'in_time_stamp': ['2022-01-01T00:00:00']

        }

    'channels_to_vname': {1: z_500, 2:z_850,...., 69:msl}
        
    save_path:  'pangu'
    
    s3_client:      
            endpoint='http://10.140.31.254'),
            
        s3_client(*s3_cfg)
    
    filter_dict =['z_1000','z_850','z_500','z_100','z_50',
                        'q_1000','q_850','q_500','q_100','q_50',
                        'u_1000','u_850','u_500','u_100','u_50',
                        'v_1000','v_850','v_500','v_100','v_50',
                        't_1000','t_850','t_500','t_100','t_50',
                        'u10', 'v10', 't2m', 'msl']    
    """


    generate_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
    Idx2ParaName = {0: 'z500', 1:'q200', 2:'u400' }
    
    if isinstance(data_sample,  Dict):
        # gt_time = str(data_sample.get('gt_time_stamp')[])
        pred_times = list(data_sample.get('pred_label').keys())
        pred_times.sort()
        
        pressure_dict = {}
        surface_dict = {}

        for channle_id, shortName_level in channels_to_vname.items():
            if shortName_level in filter_dict:
                shortName, level = shortName_level.split('_') if '_' in shortName_level else (shortName_level, None)
                if level is None:
                    if shortName not in surface_dict.keys():
                        surface_dict.update({shortName:{'idx':[]}})
                    surface_dict[shortName]['idx'].append(channle_id)
                else: 
                    if shortName not in pressure_dict.keys():
                        pressure_dict.update({shortName:{'idx':[], 'level':[]}})
                    pressure_dict[shortName]['idx'].append(channle_id)
                    pressure_dict[shortName]['level'].append(float(level))  
        initial_time = str(data_sample.get("in_time_stamp"))

        for pred_time in pred_times:
            pred = data_sample.get('pred_label').get(pred_time).squeeze(0)       # (B, C, H, W)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy() 
            _, lat_num, lon_num = pred.shape
            
            gt = None
            if 'gt_label' in data_sample.keys():
                if pred_time in data_sample.get('gt_label').keys():
                    gt = data_sample.get('gt_label').get(pred_time).clone().cpu().numpy()  
      
            # ========================== surface prediction=========================
            ds = xr.Dataset(
                {
                    ShortName: xr.DataArray(
                        data=pred[V['idx']][None],
                        dims=['time', 'isobaricInhPa', 'latitude', 'longitude'],
                        attrs={'units': grib_para[ShortName]['Unit'],
                            'GRIB_shortName': ShortName}
                        ) for ShortName, V in pressure_dict.items()
                                            
                    },
                coords={
                        'time':xr.DataArray([0],  # the lead time since  the time in attrs
                                            dims='time',
                                            attrs={"units": f"hours since {pred_time}"}
                                            ),
                        'isobaricInhPa': xr.DataArray(pressure_dict[next(iter(pressure_dict))]['level'], #list 
                                                    dims='isobaricInhPa',
                                                    attrs={
                                                            "units": "hPa", 
                                                            "long_name":"pressure",
                                                            "positive": "down",
                                                            "stored_direction":  "decreasing",
                                                            "standard_name": "air_pressure",
                                                            }
                                                ),
                        'latitude': xr.DataArray(np.linspace(90., -90., lat_num), 
                                                dims='latitude',
                                                attrs={"units": "degrees_north", 
                                                        'stored_direction': 'decreasing'}
                                                ),
                        'longitude':xr.DataArray(np.linspace(0., 360., lon_num, endpoint=False), 
                                                dims='longitude',
                                                attrs={"units": "degrees_east"}
                                                ),

                        },
                attrs={
                    "GRIB_edition": 2,
                    "description": "Prediction of AI-based NWP model: FengWu-GHR",
                    "institution": "Shanghai Ailab",
                    "contact": "Tao Han@hantao10200@gmail.com",
                    "initial_time": f'Initiai filed time: {initial_time}',
                    "history": f"First generate at:{generate_time}"
                    }
                )
            os.makedirs(f'{save_root}/{initial_time}/', exist_ok=True)
            save_path = f'{save_root}/{initial_time}/{pred_time}_pressure.grib'
            # to_grib(ds, save_path)
            ds.to_netcdf(save_path.replace('grib', 'nc'))
           
            if s3_client is not None:
                s3_uri=f'{save_root}/{initial_time[:4]}/{initial_time}/{pred_time}.nc'
                s3_client.write_nc_from_BytesIO(ds, s3_uri=s3_uri)
                print(f'upload the {s3_uri} to ceph!!!!')
                
            # ========================== surface prediction=========================

            ds = xr.Dataset(
                {
                ShortName: xr.DataArray(
                        data=pred[V['idx']],
                        dims=['time',  'latitude', 'longitude'],
                        attrs={'units': grib_para[ShortName]['Unit'],
                            'GRIB_shortName': ShortName}
                        ) for ShortName, V in surface_dict.items()
                                            
                    },
                coords={
                        'time':xr.DataArray([0],  # the lead time since  the time in attrs
                                            dims='time',
                                            attrs={"units": f"hours since {pred_time}"}
                                            ),
    
                        'latitude': xr.DataArray(np.linspace(90., -90., lat_num), 
                                                dims='latitude',
                                                attrs={"units": "degrees_north", 
                                                        'stored_direction': 'decreasing'}
                                                ),
                        'longitude':xr.DataArray(np.linspace(0., 360., lon_num, endpoint=False), 
                                                dims='longitude',
                                                attrs={"units": "degrees_east"}
                                                ),

                        },
                attrs={
                    "GRIB_edition": 2,
                    "description": "Prediction of AI-based NWP model: FengWu-GHR",
                    "institution": "Shanghai Ailab",
                    "contact": "Tao Han@hantao10200@gmail.com",
                    "initial_time": f'Initiai filed time: {initial_time}',
                    "history": f"First generate at:{generate_time}"
                    }
                )
            os.makedirs(f'{save_root}/{initial_time}/', exist_ok=True)
            save_path = f'{save_root}/{initial_time}/{pred_time}_surface.grib'
            # to_grib(ds, save_path)
            ds.to_netcdf(save_path.replace('grib', 'nc'))
            
            if s3_client is not None:
                s3_uri= f'{save_root}/single/{initial_time[:4]}/{initial_time}/{pred_time}.nc'
                s3_client.write_nc_from_BytesIO(ds, s3_uri = s3_uri)
                print(f'upload the {s3_uri} to ceph!!!!')
                
            del data_sample.get('pred_label')[pred_time]
    else:
        if isinstance(data_samples, torch.Tensor):
            data_samples = data_samples.numpy()
            
        v_num, times, level_num, lat_num, lon_num = data_samples.shape

        ds_surface = xr.Dataset(
            {
                ShortName[0]: xr.DataArray(
                    data=data_samples[idx],
                    dims=['time', 'isobaricInhPa', 'latitude', 'longitude'],
                    attrs={'units': grib_para[ShortName]['Unit'],
                            'GRIB_shortName': ShortName}
                    ) for idx, ShortName in Idx2ParaName.items()
                                            
                },
            coords={
                    'time':xr.DataArray([1],  # the lead time since  the time in attrs
                                        dims='time',
                                        attrs={"units": "hours since 2022-01-01T01"}
                                        ),
                    'isobaricInhPa': xr.DataArray([500.0, 200.0, 400.0], 
                                                    dims='isobaricInhPa',
                                                    attrs={
                                                        "units": "hPa", 
                                                        "long_name":"pressure",
                                                        "positive": "down",
                                                        "stored_direction":  "decreasing",
                                                        "standard_name": "air_pressure",
                                                        }
                                                ),
                    'latitude': xr.DataArray(np.linspace(90., -90., lat_num), 
                                                dims='latitude',
                                                attrs={"units": "degrees_north", 
                                                    'stored_direction': 'decreasing'}
                                                ),
                    'longitude':xr.DataArray(np.linspace(0., 360., lon_num, endpoint=False), 
                                                dims='longitude',
                                                attrs={"units": "degrees_east"}
                                                ),

                    },
            attrs={
                "GRIB_edition": 2,
                "description": "Forecasts of AI-based NWP model: FengWu-GHR",
                "institution": "Shanghai AILab",
                "contact": "Tao Han@hantao10200@gmail.com",
                "history": f"First generate at:{generate_time}"
            }
            )
           
        if save_path.endswith('nc'): 
            ds_surface.to_netcdf(save_path)
        elif save_path.endswith('grib'): 
            to_grib(ds_surface, save_path)
             
if __name__ == "__main__":
    data = torch.rand(3,1,3,720,1440)
    
    write_grib(data, './test_save.nc')
    write_grib(data, './test_save.grib')
