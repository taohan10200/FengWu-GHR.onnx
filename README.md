# FengWu-GHR onnx
FengWu-GHR: Learning the Kilometer-scale Medium-range Global Weather Forecasting, arXiv preprint: 2402.00059, 2023. https://arxiv.org/abs/2402.00059.



## :rocket:

Download onnx models here:

| Model |Resolution | Precision | Size | URL | Demo |
| :-: | :-: | :-: |:-: | :-: | :-: |
| FengWu-GHR (Meta model)| 721x1440| fp16 | 18GB | [Onedrive](https://pjlab-my.sharepoint.cn/:f:/g/personal/hantao_dispatch_pjlab_org_cn/EkMzQtj__wFLgyPIdAQ2DDUB-wsNhGQ80lTGX5SI20fi7w?e=MHJaRb) | [fengwu_ghr_demo.py](./fengwu_ghr_demo.py) |
| FengWu-GHR (full model) |  2001x4000 | - |- |- | - |


## News

2024/04/04 add FengWu-GHR (Meta Model) onnx-fp16 model and its demos :star:

2024/03/22 init project :star:


## Features

* Release FengWu-GHR onnx models (Meta model-only) and their onnxruntime standalone demo
* No `torch` required
* Using memory pool to dynimically dispatch merory, allowing running on small GPU memory or CUP-only devices 
* Support device: 16 GB CPU laptop/PC (very slow :turtle:) or Nvidia GPU (GPU>=20GB) 

## Getting started
### 1. Clone the code and prepare environment (if necessary) using the following command:
```bash
$ git clone https://github.com/taohan10200/FengWu-GHR.onnx.git
$ conda create -n fengwu_ghr python=3.10 -y
$ python3 -m pip install -r requirements.txt
```
### 2. Download the Pretrained ONNX model
Download `onnx` dir from [OneDrive](https://pjlab-my.sharepoint.cn/:f:/g/personal/hantao_dispatch_pjlab_org_cn/EskXFthAs-NHk42nMzoBOAcB_AfDES52Kb6RcH23OcngoQ?e=TbjSr1) and place it in the `$FengWu-GHR.onnx` (the root of this repository).

### 3. Download the `input` data from [OneDrive](https://pjlab-my.sharepoint.cn/:f:/g/personal/hantao_dispatch_pjlab_org_cn/EskXFthAs-NHk42nMzoBOAcB_AfDES52Kb6RcH23OcngoQ?e=TbjSr1) and organize it as following structure.
```
$ FengWu-GHR.onnx/
├── data/
│   ├── demos
│   ├── input
│   └── output
├── onnx/
│   └── fengwu_ghr/
│       └── meta_model/
│           ├── block_0.onnx
│           ├── ...
│           └── encoder.onnx
├── fengwu_ghr_demo.py
└── README.md
```
### 4. Run the inference and demos script
```bash
$ python -u fengwu_ghr_demo.py --timestamp=2023-06-01T00:00:00 --config=config/fengwu_ghr_cfg.py 
..
# If you only have 10 GB memory, use `--poolsize`
$ python -u fengwu_ghr_demo.py --timestamp=2023-06-01T00:00:00 --config=config/fengwu_ghr_cfg.py  --poolsize 10
..
# Try more options
$ python -u fengwu_ghr_demo.py --help
```

Reminder: After runing this script, the forecast results will be saved as `netcdf` format in `data/output/${timestamp}`. Feel free to change the  `inference_steps` and `save_cfg` in `./config/fengwu_ghr_cfg.py` for rollout length and saved variables.  
```
inference_steps = 40 # one step is 6 hour interval
save_cfg = dict(
    save_path='./data/output' ,   
    variables_list =['z_1000','z_850','z_500','z_100','z_50',
                            'q_1000','q_850','q_500','q_100','q_50',
                            'u_1000','u_850','u_500','u_100','u_50',
                            'v_1000','v_850','v_500','v_100','v_50',
                            't_1000','t_850','t_500','t_100','t_50',
                            'v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp6h', 'msl']
                )
```
## Notes
1. This demo is runing under the given input samples. If you want to inference under the forcing of other initial fields, please download them from [ERA5 pressure-level dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels) and [ERA5 single-level dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).
2. The input and forecast of `precipitation` are 6-hour accumulations. However, ERA5 provides it with houly accumulation, please handle it with care.
3. This version of FengWu-GHR (Meta Model) receives input data with a shape of `74x721x1440`, where 74 represents 74 atmospheric features. The latitude range is the `[90N, 90S]`, and the longitude range is `[-180, 180]`. The first 65 variables are pressure-level variables in the order of  `['z', 'q', 'u', 'v', 't']`, where each variable has 13 levels: `[1000.,  925.,  850.,  700.,    600.,   500.,   400., 300.,  250.,    200.,   150.,    100., 50.]`. The remain 9 variable are surface variables:`['v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp6h', 'msl']`. Refer to [fengwu_ghr_cfg.py](./config/fengwu_ghr_cfg.py) for more information.




## Demos
The following are demos of 10-days lead forecasting after running above script:

### Demo 1: Some Surface Variables: `10 mete wind (v10), 100 meter wind (v100), 2 meter temperature (t2m), surface pressure (sp), 6-hour precipitation (tp6h), mean sea-level pressure (msl)`
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