import os
import argparse
import importlib.util
import sys
import cdsapi
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd
import copy
import xarray as xr

# Disable HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def format_size(bytes, unit='GB'):
    """Format file size in KB, MB, or GB"""
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except ValueError:
        print("The 'bytes' parameter is not valid")
        return "Error"

    if unit == "GB":
        return kb / 1024 / 1024
    elif unit == "MB":
        return kb / 1024
    else:
        return kb


class ERA5Downloader:
    """
    ERA5 data downloader class.
    Handles downloading, saving, and processing data from the Copernicus Climate Data Store (CDS).
    """

    def __init__(self, config_path):
        # Dynamically load configuration file
        if importlib.util.find_spec("mmengine") is not None and sys.version_info >= (3, 0):
            from mmengine import Config
        else:
            from mmcv import Config

        self.cfg = Config.fromfile(config_path)
        self.proxies = self._setup_environment()

        self.local_root = self.cfg.storage.local
        self.ecmwf_dataset_pressure = 'reanalysis-era5-pressure-levels'
        self.ecmwf_dataset_single = 'reanalysis-era5-single-levels'

        # Initialize CDSAPI client
        self.cdsapi_client = cdsapi.Client(
            url=os.environ.get("CDSAPI_URL"),
            key=os.environ.get("CDSAPI_KEY")
        )
        self.cdsapi_client.session.proxies.update(self.proxies)

        # Configure request dictionaries
        self.pressure_request_dic = copy.deepcopy(self.cfg.pressure_request_dic)
        self.single_request_dic = copy.deepcopy(self.cfg.single_request_dic)
        self.accumulation_request_dic = copy.deepcopy(self.cfg.accumulation_request_dic)
    def _setup_environment(self):
        """
        Set up the environment variables and proxy settings.
        """
        os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
        os.environ['CDSAPI_KEY'] = 'ea3a2607-158c-48a4-bd27-b255256b2759'

        proxy_type = self.cfg.proxy.type

        if proxy_type == 'direct':
            print('No proxy is used.')
            return {}

        elif proxy_type == 'normal':
            print(f'Using normal proxy: {self.cfg.proxy.normal}')
            return dict(http=self.cfg.proxy.normal, https=self.cfg.proxy.normal)

        elif proxy_type == 'special':
            print(f'Using special proxy: {self.cfg.proxy.special}')
            return dict(http=self.cfg.proxy.special, https=self.cfg.proxy.special)

        else:
            raise ValueError("Proxy type must be 'direct', 'normal', or 'special'.")

    def save(self, time_required, file_key):
        """
        Save ERA5 data by downloading pressure-level and single-level datasets.
        """
        prefix, extension = os.path.splitext(file_key)

        # Download pressure-level data
        request_dic = copy.deepcopy(self.pressure_request_dic)
        request_dic.update(time_required)
        pressure_file = f'{prefix}_pressure{extension}'

        print(f'Checking and downloading pressure file: {pressure_file}')
        if not self._check_filesize(request_dic, self.ecmwf_dataset_pressure, pressure_file):
            self._download_file(self.ecmwf_dataset_pressure, request_dic, pressure_file)

        # Download single-level data
        request_dic = copy.deepcopy(self.single_request_dic)
        request_dic.update(time_required)
        single_file = f'{prefix}_single_base{extension}'

        print(f'Checking and downloading single file: {single_file}')
        if not self._check_filesize(request_dic, self.ecmwf_dataset_single, single_file):
            self._download_file(self.ecmwf_dataset_single, request_dic, single_file)

        # Process time-series data
        start_time = datetime.strptime(
            f"{time_required['year']}-{time_required['month']}-{time_required['day']} {time_required['time']}",
            "%Y-%m-%d %H:%M:%S"
        )
        ds_base = xr.open_dataset(single_file, engine='netcdf4')
        ds_base.load()
        os.remove(single_file)

        # Download accumulation data
        request_dic = copy.deepcopy(self.accumulation_request_dic)
        request_dic.update(time_required)

        ds_acc = self._process_time_series(start_time, request_dic, prefix, extension)
    
        
        # Merge the datasets
        ds_acc = ds_acc.rename({"tp": "tp6h"})
        ds_acc = ds_acc.rename({"ssr": "ssr6h"})
        # import pdb
        # pdb.set_trace()    
        ds = xr.merge([ds_base, ds_acc])
        ds.to_netcdf(f"{prefix}_single{extension}")
        
    def _process_time_series(self, start_time, request_dic, prefix, extension):
        """
        Process time-series data by downloading additional time offsets and calculating summaries.
        """
        acc_list = []
        for i in range(0, 6):
            time_offset = start_time - timedelta(hours=i)
            yy, mm, dd, hh = self._get_ymdh(time_offset)

            past_time_required = dict(year=yy, month=mm, day=dd, time=hh)
            request_dic.update(past_time_required)

            temp_file = f"{prefix}_acc{i}h{extension}"
            if not self._check_filesize(request_dic, self.ecmwf_dataset_single, temp_file):
                self._download_file(self.ecmwf_dataset_single, request_dic, temp_file)

            ds_temp = xr.open_dataset(
                    temp_file,
                    engine="netcdf4",
                )
            #load to memory 
            ds_temp.load()
            acc_list.append(ds_temp.copy())
           
            os.remove(temp_file)
        # [print(i['ssr'].data.mean()) for i in acc_list]
        # import pdb
        # pdb.set_trace()    
        assert "valid_time" in  ds_temp.keys()
        combined_acc = xr.concat(acc_list, dim="valid_time")
        summarized_acc = combined_acc.sum(dim="valid_time")
        return summarized_acc

    def _download_file(self, dataset, request_dic, file_name):
        """
        Download a file from the CDSAPI.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.cdsapi_client.retrieve(dataset, request_dic, file_name)

    def _check_filesize(self, request_dic, dataset, file_name):
        """
        Check if the local file size matches the remote file size.
        """
        local_size = os.path.getsize(file_name) if os.path.exists(file_name) else 0
        remote_size = self.cdsapi_client.retrieve(dataset, request_dic).content_length

        if local_size == remote_size:
            print(f"File {file_name} is complete: {local_size} bytes.")
            return True
        else:
            print(f"File {file_name} is incomplete: {local_size} != {remote_size}.")
            return False

    @staticmethod
    def _get_ymdh(timestamp):
        """
        Get year, month, day, and hour from a timestamp.
        """
        timestamp = pd.to_datetime(timestamp)
        return (str(timestamp.year).zfill(4),
                str(timestamp.month).zfill(2),
                str(timestamp.day).zfill(2),
                str(timestamp).split(' ')[-1])

    def get_from_timestamp(self, time_stamp, local_root=None):
        """
        Download data for a specific timestamp.
        """
        yy, mm, dd, hh = self._get_ymdh(time_stamp)
        time_required = dict(year=yy, month=mm, day=dd, time=hh)
        local_root = local_root or self.local_root
        time_stamp = str(time_stamp).replace(' ', 'T')
        file_path = f"{local_root}/{yy}/{time_stamp}.nc"


        self.save(time_required, file_path)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="ERA5 Data Downloader")
    parser.add_argument('--st', type=str, required=True, help='Start timestamp (ISO format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--et', type=str, default='', help='End timestamp (ISO format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--local_root', type=str, default='./data/input/era5', help='Path to save the ERA5 data')
    args = parser.parse_args()

    # Initialize ERA5Downloader with configuration
    downloader = ERA5Downloader('config/era5_config.py')

    # Define time range for data downloads
    end_time = args.et or args.st
    time_stamps = pd.date_range(start=args.st, end=end_time, freq='6H')

    # Download data for each timestamp
    for timestamp in time_stamps:
        downloader.get_from_timestamp(time_stamp=timestamp, local_root=args.local_root)