import cfgrib
import pandas as pd
import xarray as xr 
import numpy as np
from scipy.ndimage import zoom


# define the time range 
start_time = pd.Timestamp("2023-07-01 00:00:00")
# end_time = pd.Timestamp("2024-01-02 00:00:00")

# read pressure-level grib
# dataset = xr.open_dataset('./data/input/2023-07-01.grib', engine='cfgrib')

# split_datasets = dataset.groupby('time')

# for time, split_data in split_datasets:
#     import pdb
#     pdb.set_trace()
#     split_data.to_netcdf(f"./data/input/{str(time).split('.')[0]}_pressure.nc")

    
# dataset.close()

# read precipitation grib
dataset = xr.open_dataset(
                        './data/input/202307.grib', 
                        engine='cfgrib', 
                        ackend_kwargs={'indexpath': ''}
                        )

split_datasets = dataset.groupby('time')

for time, split_data in split_datasets:
    # import pdb
    # pdb.set_trace()
    
    # split_data.to_netcdf(f"./data/input/{str(time).split('.')[0]}_precipitation.nc")
    data = split_data['tp'].data*1000
    print(f'time:{time}, mean: {data.mean()}, max:{data.max()}, min:{data.min()}')
    
dataset.close()


# read precipitation grib
dataset = xr.open_dataset('./data/input/2023-06-30.nc', engine='netcdf4')

split_datasets = dataset.groupby('time')

acc_data =  np.zeros((721, 1440), dtype=np.float32)
for time, split_data in split_datasets:
    # import pdb
    # pdb.set_trace()
    
    # split_data.to_netcdf(f"./data/input/{str(time).split('.')[0]}_precipitation.nc")
    data = split_data['tp'].data*1000
    acc_data += data
    print(f'ori_data:{time}, mean: {acc_data.mean()}, max:{acc_data.max()}, min:{acc_data.min()}')
    
    zoom_data = zoom(acc_data, zoom=(2001/721, 4000/1440), order=3)
    print(zoom_data.shape)
    print(f'zoom_data:{time}, mean: {acc_data.mean()}, max:{acc_data.max()}, min:{acc_data.min()}')
    
dataset.close()
