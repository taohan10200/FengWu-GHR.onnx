# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import os
import argparse
import json
import torch
import numpy as np
import xarray as xr
from loguru import logger
from mmengine.config import Config, DictAction
from datetime import datetime, timedelta
from tools.write_to_grib import write_grib
from scipy.ndimage import zoom
import torch.nn.functional as F
import time
from nwp_models.fengwu_ghr_lora import FengWu_Hres_Lora



class FengWu_GHR_Inference:
    def __init__(self, cfg: dict = {}):
        self.cfg = cfg
        self.dataset = self.cfg.dataset
        self.output_root = os.path.join(self.cfg.save_cfg.save_path, self.dataset)
        checkpoint_dir = cfg.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            logger.error('{} not exist'.format(checkpoint_dir))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'The inference is performed on {self.device}')
        # pool = MemoryPoolSimple(cfg.poolsize_GB)
        self.model = FengWu_Hres_Lora(**cfg.backbone).to(self.device).eval()
        self.load_state_dict()
        if self.cfg.fp16:
            self.model.half()


        self.level_mapping =  [cfg.total_levels.index(val) for val in cfg.pressure_level if val in cfg.total_levels ]
        self.mean, self.std = self.get_mean_std() #read the channel-wise mean and std according to the defined variable in configuration.
        self.mean = torch.tensor(self.mean[None,:,None,None]).to(self.device)
        self.std =  torch.tensor(self.std[None,:,None,None]).to(self.device)
        
        self.channels_to_vname, self.vname_to_channels = self.channel_vname_mapping()
                
        self.input_shape = cfg.input_shape
        
    def load_state_dict(self):
        pretrained_dict = torch.load(self.cfg.checkpoint_dir, map_location='cpu'
                                )['state_dict']
        model_dict = self.model.state_dict()

        pretrained_dict_filter ={}
        for k, v in pretrained_dict.items():
            if k[9:] in model_dict.keys():
                pretrained_dict_filter.update({k[9:]: v})
        print(
                "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict_filter)
                                               )))
        model_dict.update(pretrained_dict_filter)

        self.model.load_state_dict(model_dict, strict=True)
        
        
    def get_mean_std(self):
        with open('./nwp_models/mean_std.json',mode='r') as f:
            mean_std = json.load(f)
            f.close()
        with open('./nwp_models/mean_std_single.json',mode='r') as f:
            mean_std_single = json.load(f)
            f.close()
        mean_list, std_list = [],[]
        for  vname in self.cfg.vnames.get('pressure'):
            mean_list += [mean_std['mean'][vname][idx] for idx in self.level_mapping]
            std_list += [mean_std['std'][vname][idx] for idx in self.level_mapping]
        for vname in self.cfg.vnames.get('single'):
            mean_list.append(mean_std_single['mean'][vname])
            std_list.append(mean_std_single['std'][vname])
            
        return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)
        

    def convert_to_fp16(self, inputs):
        if inputs.dtype == np.float32:
            inputs = inputs.astype(np.float16)
        elif inputs.dtype == torch.float32:
            inputs = inputs.to(torch.float16)
            
        return inputs

    def inference(self, 
                  timestamp:str):
        """_summary_

        Args:
            timestamp: the timestamp of the initial field. Defaults to str.

        Returns:
            outputs: 
        """
        if self.dataset == 'era5':
            inputs = self.reada_era5_initial_field(timestamp) #4D input: [batch_size, vname_number, H, W]
        else:
            inputs = self.read_analysis_initial_field(timestamp)#4D input: [batch_size, vname_number, H, W]
        outputs = inputs
 
        in_time_stamp = timestamp
        
        datasample_initial = {
                     'pred_label':{in_time_stamp:self.de_normalization(inputs.clone().float())},
                     'in_time_stamp':in_time_stamp
                    }
        write_grib(datasample_initial, 
                   save_root = self.output_root, 
                   channels_to_vname = self.channels_to_vname, 
                   filter_dict = self.cfg.save_cfg.variables_list)
 
        for step in range(self.cfg.inference_steps):   
            if self.cfg.fp16:
                outputs = self.convert_to_fp16(outputs)
            
            x = {'input':outputs, 
                 'step': np.array(step, dtype=np.int64)}
            
            st = time.time()
            with torch.no_grad():
                outputs = self.model(x['input'], step = 0)#x['step'])
           
            et1 = time.time()
            new_dt = datetime.fromisoformat(timestamp) + timedelta(hours=6)
            timestamp = new_dt.isoformat()
            
            outputs_ = self.de_normalization(outputs.clone().float())
            
            print(f'Step: {step}, Initial time: {in_time_stamp}, forecast time: {timestamp}, inference spend: {(et1-st):.2f}s')


            datasample = {
                            'pred_label':{timestamp:self.process_output(outputs_)},
                            'in_time_stamp':in_time_stamp
                          }

            et2 = time.time()

            print(f' Post processing time is: {(et2-et1):.2f}s')
            
            # if step == 39:
            write_grib(datasample, 
                    save_root = self.output_root, 
                    channels_to_vname=self.channels_to_vname, 
                    filter_dict = self.cfg.save_cfg.variables_list)
        
            print(f' forecasts are saved at {self.output_root} with nc format , save time is: {(et2-et1):.2f}s')

    
    def reada_era5_initial_field(self, timestamp):
        input_initial_field=[]
        try:
            pressure_data = xr.open_dataset(f'./data/input/era5/{timestamp[:4]}/{timestamp}_pressure.nc', 
                                        engine='netcdf4',
                      )
            surface_data = xr.open_dataset(f'./data/input/era5/{timestamp[:4]}/{timestamp}_single.nc', 
                            engine='netcdf4',
        )
        except Exception as e:
            print("An error occurred:", e)
            raise SystemExit("Program terminated due to an error.")

        for vname in self.cfg.vnames.get('pressure'):
            vname_data = pressure_data[vname]
            for height in self.cfg.pressure_level:
                vdata = vname_data.sel(level=height).data
                vdata = vdata.squeeze() if vdata.ndim>=3 else vdata
                vdata = torch.tensor(vdata[None,None,:,:]).to(self.device)
                vdata = self.check_input(vdata)
                assert  vdata.ndim==4
                input_initial_field.append(vdata)

        for vname in self.cfg.vnames.get('single'):
            vdata = surface_data[vname].data
            vdata = vdata.squeeze() if vdata.ndim>=3 else vdata
            vdata = torch.tensor(vdata[None,None,:,:]).to(self.device)
            vdata = self.check_input(vdata)
            assert  vdata.ndim==4
            if 'tp' in vname:
                vdata =  vdata*1000  # if the unit is meter, please transfer it to millmeter
            input_initial_field.append(vdata)
        
        input_initial_field = torch.cat(input_initial_field, axis=0)
        
        return self.normalization(input_initial_field)
            
    def read_analysis_initial_field(self, timestamp):
        input_initial_field=[]
        try:
            data = xr.open_dataset(f'./data/input/analysis/{timestamp}.grib')
        except Exception as e:
            print("An error occurred:", e)
            raise SystemExit("Program terminated due to an error.")
 
        for vname in self.cfg.vnames.get('pressure'):
            vname_data = data[vname]
            for height in self.cfg.pressure_level:
                vdata = vname_data.sel(isobaricInhPa=height).data
                vdata = torch.tensor(vdata[None,None,:,:]).to(self.device)
                vdata = self.check_input(vdata)
                input_initial_field.append(vdata)

        for vname in self.cfg.vnames.get('single'):
            vdata = data[vname].data
            vdata = torch.tensor(vdata[None,None,:,:]).to(self.device)
            vdata = self.check_input(vdata)

            if 'tp' in vname:
                vdata =  vdata*1000  # if the unit is meter, please transfer it to millmeter

            input_initial_field.append(vdata)
        
        input_initial_field = torch.cat(input_initial_field, axis=1)
        return self.normalization(input_initial_field)
    
    def check_input(self, vdata):
        vdata = F.interpolate( vdata, size=self.input_shape, mode='bicubic')
        return vdata
    
    def process_output(self, outputs_):
        from scipy.ndimage import uniform_filter,gaussian_filter 
        def apply_gaussian_filter(tensor, sigma):
            # Convert tensor to numpy array for filtering
            tensor_np = tensor.cpu().numpy()
            
            # Apply Gaussian filter only on the last two dimensions
            filtered_np = gaussian_filter(tensor_np, sigma=(0, 0, sigma, sigma))
            
            # Convert back to torch tensor
            filtered_tensor = torch.from_numpy(filtered_np)
            
            return filtered_tensor
        
        outputs_ = apply_gaussian_filter(outputs_, sigma=1.5)
        no_less_than_zero = ['tp6h']
        for i in no_less_than_zero:
            idx = self.vname_to_channels.get(i)
            outputs_[:, idx, :, :][outputs_[:, idx, :, :] < 0] = 0
        return outputs_
    
    def normalization(self, data):
        data -= self.mean
        data /= self.std
        return data
    
    def de_normalization(self, data):
        data *= self.std
        data += self.mean
        return data   
    
    def channel_vname_mapping(self):
        channels_to_vname={}
        vname_to_channels={}
        ch_idx = 0
        for v in self.cfg.vnames.get('pressure'):
            for level in self.cfg.pressure_level:
                channels_to_vname.update({ch_idx: v+'_'+str(int(level)) })
                vname_to_channels.update({v+'_'+str(int(level)): ch_idx })
                ch_idx += 1
        for v in self.cfg.vnames.get('single'):
            channels_to_vname.update({ch_idx: v })
            vname_to_channels.update({v: ch_idx})
            ch_idx += 1
        return channels_to_vname, vname_to_channels
         
    
def parse_args():
    parser = argparse.ArgumentParser(description='FengWu-GHR.onnx onnxruntime demo')
    parser.add_argument(
                        '--config',
                        type=str,
                        help='inference config file path')
    
    parser.add_argument('--onnxdir',
                        default='onnx/fengwu_ghr/meta_model',
                        type=str,
                        help='fengwu-ghr onnx model directory.')
    parser.add_argument('--poolsize',
                        default=None,
                        type=int,
                        help='The size of cpu/gpu memory allocated for inference.')
    parser.add_argument('--timestamp',
                        default='2024-07-08T18:00:00',
                        type=str,
                        help='The timestamp of the initial field.')
    parser.add_argument('--gpu',
                        default=None,
                        type=int,
                        help='The timestamp of the initial field.'
                        )
    parser.add_argument('--dataset',
                        default='analysis',
                        type=str,
                        help='The datasource of the initial field. Both ERA5 and analysis from EC are supported.'
                        )
    parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space '
            'is allowed.')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIABLE']=str(args.gpu)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    if args.poolsize is not None:
        cfg.poolsize_GB = args.poolsize
        
    if args.dataset is not None:
        cfg.dataset = args.dataset   
    logger.warning(args)
    
    FengWu_GHR = FengWu_GHR_Inference(
                    cfg=cfg,
                    )
    
    FengWu_GHR.inference(args.timestamp)


if __name__ == '__main__':
    main()
