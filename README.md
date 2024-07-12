# FengWu-GHR onnx

!!! Note: Open source is not easy, please star to show support.
## :rocket: Reading paper first:
###  FengWu-GHR: Learning the Kilometer-scale Medium-range Global Weather Forecasting, arXiv preprint: 2402.00059, 2023. https://arxiv.org/abs/2402.00059.

We now support two version of FengWu-GHR:

| Model |Resolution | Precision | Size | URL | Demo |
| :-: | :-: | :-: |:-: | :-: | :-: |
| FengWu-GHR (meta_model_0.25°)| 721x1440, 0.25°| fp16 | 9.0GB | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/thanad_connect_ust_hk/EYY_VIxLltlMvkqG-1T6IBEBXYcPWHF5PwKrUL2TJfTt2g?e=pQbJHo) | [fengwu_ghr_inference_25km.py](./fengwu_ghr_inference_25km.py) |
| FengWu-GHR (GHR_0.09° ) |  2001x4000, 0.09° | fp16 |6.5GB |[Request via email](mailto:hantao10200@gmail.com) | [fengwu_ghr_inference_9km.py](./fengwu_ghr_inference_9km.py) |


## News
2024/07/12 add FengWu-GHR (full model) onnx-fp16 model :star:

2024/04/04 add FengWu-GHR (Meta Model) onnx-fp16 model and its demos :star:

2024/03/22 init project :star:


## Features

* Release FengWu-GHR onnx models (Meta model-only) and their onnxruntime standalone demo
* No `torch` required
* Using memory pool to dynimically dispatch merory, allowing running on small GPU memory or CUP-only devices 
* Support device: 16 GB CPU laptop/PC (very slow :turtle:) or Nvidia GPU (GPU>=20GB for meta model and GPU>50GB for GHR_0.09) 

## Getting started
### 1. Clone the code and prepare environment (if necessary) using the following command:
```bash
$ git clone https://github.com/taohan10200/FengWu-GHR.onnx.git
$ conda create -n fengwu_ghr python=3.10 -y
$ source activate fengwu_ghr
$ python3 -m pip install -r requirements.txt
```
### 2. Download the Pretrained ONNX model.
Download [meta_model_0.25](https://hkustconnect-my.sharepoint.com/:u:/g/personal/thanad_connect_ust_hk/EYY_VIxLltlMvkqG-1T6IBEBXYcPWHF5PwKrUL2TJfTt2g?e=pQbJHo) and unzip it in `$FengWu-GHR.onnx/onnx/` (the root of this repository).

### 3. Get the input data/ initial field
Download sample input from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/thanad_connect_ust_hk/EmLxeXdC2a5NlJ0RW5ZHyvEBCuFcABtk5Z3SqbIGwEyK2w?e=8ZKuzU).


> We support the grib format input, which  waives a complex data prepareness. What you should do is to organize your data as a packed grib file after you getting the initial field for ECMWF or other data sources. Below is a sample we provided:  

```python
import xarray as xr
input_data = xr.open_dataset('./data/input/analysis/2024-07-08T18:00:00.grib')
input_data
Out[1]: 
<xarray.Dataset>
Dimensions:        (latitude: 1801, longitude: 3600, isobaricInhPa: 13)
Coordinates:
  * latitude       (latitude) float32 90.0 89.9 89.8 89.7 ... -89.8 -89.9 -90.0
  * longitude      (longitude) float32 0.0 0.1 0.2 0.3 ... 359.7 359.8 359.9
  * isobaricInhPa  (isobaricInhPa) int32 1000 925 850 700 600 ... 200 150 100 50
Data variables: (12/16)
    v10            (latitude, longitude) float32 ...
    u10            (latitude, longitude) float32 ...
    v100           (latitude, longitude) float32 ...
    u100           (latitude, longitude) float32 ...
    t2m            (latitude, longitude) float32 ...
    tcc            (latitude, longitude) float32 ...
    ...             ...
    z              (isobaricInhPa, latitude, longitude) float32 ...
    q              (isobaricInhPa, latitude, longitude) float32 ...
    u              (isobaricInhPa, latitude, longitude) float32 ...
    v              (isobaricInhPa, latitude, longitude) float32 ...
    t              (isobaricInhPa, latitude, longitude) float32 ...
    w              (isobaricInhPa, latitude, longitude) float32 ...
```

**Note**: The requirement for `tp6h` in the initial field is the accumulated precipitation over the past six hours from the analysis time. It can be derived from the predictions intilized at the last time.

> If you are with diffculties to get the high-reslolution analysis data. We here also provide a  portable way to download the EAR5 data as initial field. 

```python
ython tools/era5_downloader.py --time_stamp='2024-07-01T00:00:00' --local_root='./data/input/era5'
```
### 4.  Organize your project as following structure.
```
$ FengWu-GHR.onnx/
├── data/
│   ├── demos
│   ├── input
│   └── output
├── onnx/   
│     └── meta_model_0.25/
│           ├── block_0.onnx
│           ├── ...
│           └── encoder.onnx
│     └── ghr_0.09/
├── fengwu_ghr_inference_9km.py
├── fengwu_ghr_inference_25km.py
├── LICENSE
└── README.md

```
### 5. Run the inference and demos script
```bash
## Inference for high resolution forecast: 0.09x0.09 
$ python -u fengwu_ghr_inference_9km.py --timestamp=2024-07-08T18:00:00  --config=config/config/fengwu_ghr_cfg_74v_0.09.py 

## Inference for high resolution forecast: 0.25x0.25 
# If you only have 10 GB memory, use `--poolsize`
python -u fengwu_ghr_inference_25km.py --timestamp=2024-07-01T00:00:00 --config=config/fengwu_ghr_cfg_74v_0.25.py  --poolsize 10

# Try more options
$ python -u fengwu_ghr_inference.py --help
```

Reminder: After runing this script, the forecast results will be saved as `netcdf` format in `data/output/${timestamp}`. Feel free to change the  `inference_steps` and `save_cfg` in [fengwu_ghr_cfg.py](./config/fengwu_ghr_cfg.py) for rollout length and saved variables.  
```
inference_steps = 40 # one step is 6 hour interval
save_cfg = dict(
    save_path='./data/output' ,   
    variables_list =[
        'z_1000','z_850','z_500','z_100','z_50',
        'q_1000','q_850','q_500','q_100','q_50',
        'u_1000','u_850','u_500','u_100','u_50',
        'v_1000','v_850','v_500','v_100','v_50',
        't_1000','t_850','t_500','t_100','t_50',
        'v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp6h', 'msl'
        ]
    )
```
### 5. Notes
1. This demo is runing under the given input samples. If you want to inference under the forcing of other initial fields, please download them from [ERA5 pressure-level dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels) and [ERA5 single-level dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).
2. The input and forecast of `precipitation` are 6-hour accumulations. However, ERA5 provides it with hourly accumulation, please process it with care if you are using your own data.
3. This version of FengWu-GHR (Meta Model) receives input data with a shape of `74x721x1440`, where 74 represents 74 atmospheric features. The latitude range is the `[90N, 90S]`, and the longitude range is `[-180, 180]`. The first 65 variables are pressure-level variables in the order of  `['z', 'q', 'u', 'v', 't']`, where each variable has 13 levels: `[1000.,  925.,  850.,  700.,    600.,   500.,   400., 300.,  250.,    200.,   150.,    100., 50.]`. The remain 9 variable are surface variables:`['v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp6h', 'msl']`. Refer to [fengwu_ghr_cfg.py](./config/fengwu_ghr_cfg.py) for more information.




## Demos
The following are demos of 10-days lead forecasting after running above script:

### Demo 1: Some Surface Variables: `10 meter wind (v10), 100 meter wind (v100), 2 meter temperature (t2m), surface pressure (sp), 6-hour precipitation (tp6h), mean sea-level pressure (msl)`
![surface](./data/demos/surface_forecast_vs_real.gif)

### Demo 2: Geopotential
![z](./data/demos/z_forecast_vs_real.gif)

### Demo 3: Specific Humidity
![q](./data/demos/q_forecast_vs_real.gif)

### Demo 4: U Component of Wind
![u_wind](./data/demos/u_forecast_vs_real.gif)


### Demo 5: V Component of Wind
![v_wind](./data/demos/v_forecast_vs_real.gif)

### Demo 6: Temperature
![t](./data/demos/t_forecast_vs_real.gif)




<!-- ## Acknowlegements
* [RWKV](https://github.com/BlinkDL/ChatRWKV)
* [LLaMa](https://github.com/facebookresearch/llama)
* [alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* [alpaca-lora](https://github.com/tloen/alpaca-lora)
* [transformers](https://github.com/huggingface/transformers)
* [peft](https://github.com/huggingface/peft)
* [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
 -->



## License and attribution 
The copyright of this project belongs to [Shanghai Artificial Intelligence Laboratory](https://www.shlab.org.cn/), and the commercial use of these models is forbidden.


The code are licensed under the Apache License, Version 2.0. You may obtain a copy of the License at: https://www.apache.org/licenses/LICENSE-2.0.

The model weights are made available for use under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). You may obtain a copy of the License at: https://creativecommons.org/licenses/by-nc-sa/4.0/.

The weights were trained on ECMWF's ERA5 and HRES data. The repo includes a few examples of ERA5 and HRES data that can be used as inputs to the models. ECMWF data product are subject to the following terms:

 1. Copyright statement: Copyright "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)".
 2. Source www.ecmwf.int
 3. Licence Statement: ECMWF data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0). 
https://creativecommons.org/licenses/by/4.0/
 4. Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.

## Citation
```
@article{han2024fengwughr,
title={FengWu-GHR: Learning the Kilometer-scale Medium-range Global Weather Forecasting}, 
author={Tao Han and Song Guo and Fenghua Ling and Kang Chen and Junchao Gong and Jingjia Luo and Junxia Gu and Kan Dai and Wanli Ouyang and Lei Bai},
year={2024},
eprint={2402.00059},
archivePrefix={arXiv},
primaryClass={cs.LG}
}
``````