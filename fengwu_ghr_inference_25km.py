# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import os
import argparse
import json
import numpy as np
import xarray as xr
from loguru import logger
from nwp_models import FengWu_GHR
from nwp_models import MemoryPoolSimple, npsoftmax, npmultinominal2D
from mmengine.config import Config, DictAction
from datetime import datetime, timedelta
from tools.write_to_grib import write_grib
from tools.plot_demo_gif import plot_pressure_demo_gif, plot_surface_demo_gif



class FengWu_GHR_Inference:
    def __init__(self, args, cfg: dict = {}):
        self.cfg = cfg
        onnxdir = cfg.onnx_dir
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))
        assert os.path.isdir(onnxdir)

        pool = MemoryPoolSimple(cfg.poolsize_GB)
        self.model = FengWu_GHR(pool, onnxdir, cfg.onnx_keys)

        
        self.level_mapping =  [cfg.total_levels.index(val) for val in cfg.pressure_level if val in cfg.total_levels ]
        self.mean, self.std = self.get_mean_std() #read the channel-wise mean and std according to the defined variable in configuration.
        self.channels_to_vname = self.get_meta_info()

        
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
        return inputs

    def inference(self, 
                  timestamp:str, 
                  rollout_time: str):
        """_summary_

        Args:
            timestamp: the timestamp of the initial field. Defaults to str.
            rollout_time: time length of extrapolation. Defaults to None.

        Returns:
            outputs: 
        """
        # embed space
        inputs = self.read_initial_field(timestamp)[None,:,:,:] #4D input: [batch_size, vname_number, H, W]

        outputs = inputs
        prediction = {}
        in_time_stamp = timestamp
        
        
        datasample_initial = {
                     'pred_label':{in_time_stamp:self.de_normalization(np.copy(inputs.astype(np.float32)))},
                     'in_time_stamp':in_time_stamp
                    }
        write_grib(datasample_initial, 
                   save_root = self.cfg.save_cfg.save_path, 
                   channels_to_vname = self.channels_to_vname, 
                   filter_dict = self.cfg.save_cfg.variables_list)
        
        for step in range(self.cfg.inference_steps):
           
            if self.cfg.fp16:
                outputs = self.convert_to_fp16(outputs)
                
            x = {'input':outputs, 
                 'step': np.array(step, dtype=np.int64)}
            
            outputs = self.model.one_step(x)
            outputs = outputs['output']
            
            new_dt = datetime.fromisoformat(timestamp) + timedelta(hours=6)
            timestamp = new_dt.isoformat()

            prediction.update({timestamp: self.de_normalization(np.copy(outputs.astype(np.float32)))})


            datasample = {
                            'pred_label':prediction,
                            'in_time_stamp':in_time_stamp
                          }

            write_grib(datasample, 
                       save_root = self.cfg.save_cfg.save_path, 
                       channels_to_vname=self.channels_to_vname, 
                       filter_dict = self.cfg.save_cfg.variables_list)
            
        return prediction

    def read_initial_field(self, timestamp):
        input_initial_field=[]
        try:
            pressure_data = xr.open_dataset(f'./data/input/era5/{timestamp[:4]}/{timestamp}_pressure.nc', 
                                        engine='cfgrib',
                                        backend_kwargs={'indexpath': ''})
            surface_data = xr.open_dataset(f'./data/input/era5/{timestamp[:4]}/{timestamp}_single.nc', 
                            engine='cfgrib',
                            backend_kwargs={'indexpath': ''})
        except Exception as e:
            print("An error occurred:", e)
            raise SystemExit("Program terminated due to an error.")
      

        for vname in self.cfg.vnames.get('pressure'):
            vname_data = pressure_data[vname]
            for height in self.cfg.pressure_level:
                data = vname_data.sel(isobaricInhPa=height).data
                input_initial_field.append(data[None,:,:])

        for vname in self.cfg.vnames.get('single'):
            # try:
            #     data = np.load(f'./data/input/era5/{timestamp}/{vname}.npy')
            # except Exception as e:
            #     print("An error occurred:", e)
            #     raise SystemExit("Program terminated due to an error.")
            data = surface_data[vname].data
            if vname == 'tp':
                data =  data*1000  # if the unit is meter, please transfer it to millmeter
            input_initial_field.append(data[None,:,:])
        
        input_initial_field = np.concatenate(input_initial_field, axis=0)
        
        return self.normalization(input_initial_field)
            
    def normalization(self, data):
        data -= self.mean[:,np.newaxis,np.newaxis]
        data /= self.std[:,np.newaxis,np.newaxis]
        return data
    
    def de_normalization(self, data):
        data *= self.std[np.newaxis,:,np.newaxis,np.newaxis]
        data += self.mean[np.newaxis,:,np.newaxis,np.newaxis]
        return data   
    
    def get_meta_info(self):
        channels_to_vname = {}
        ch_idx = 0
        for v in self.cfg.vnames.get('pressure'):
            for level in self.cfg.pressure_level:
                channels_to_vname.update({ch_idx: v+'_'+str(int(level)) })
                ch_idx += 1
        for v in self.cfg.vnames.get('single'):
            channels_to_vname.update({ch_idx: v })
            ch_idx += 1
        return channels_to_vname
         
    
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
    parser.add_argument('--warning_gpu_memory',
                        default=None,
                        type=int,
                        help='The size of cpu/gpu memory allocated for inference.')
    parser.add_argument('--max_gpu_memory',
                        default=None,
                        type=80,   #unit GB
                        help='The maximum of gpu memory for your device.')
    parser.add_argument('--timestamp',
                        default='2023-06-01T00:00:00',
                        type=str,
                        help='The timestamp of the initial field.')
    parser.add_argument('--gpu',
                        default=None,
                        type=int,
                        help='The timestamp of the initial field.'
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
         
    logger.warning(args)
    FengWu_GHR = FengWu_GHR_Inference(
                    args=args,
                    cfg=cfg
                    )
    # np.random.seed(42)
    # data = np.random.rand(1,74, 721, 1440).astype(np.float16)
    # data = np.ones((1,74, 721, 1440)).astype(np.float16)
    # data = np.random.rand(1,7200, 3072).astype(np.float16)
    
    FengWu_GHR.inference(args.timestamp, np.array(0, dtype=np.int64))
    
    # plot_pressure_demo_gif(initial_timestamp=args.timestamp, 
    #               steps=40, 
    #               plot_variable={'z':[850,500],
    #                              'q':[850,500],
    #                              'u':[850,500],
    #                              'v':[850,500],
    #                              't':[850,500],
    #                              }
    #                 )
    
    # plot_surface_demo_gif(initial_timestamp=args.timestamp, 
    #               steps=40, 
    #               plot_variable=['sp','v10','v100', 't2m','tp6h', 'msl']
    #                             )

if __name__ == '__main__':
    main()
