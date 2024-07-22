# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
from loguru import logger
from .utils import singleton
import onnxruntime as ort
import numpy as np
import os
import psutil
import math
import torch
import pynvml

def get_mem_info():
    """
    Get GPU memory usage information based on the GPU ID, in MB units.
    :param gpu_id: GPU ID
    :return: total - total GPU memory, used - currently used GPU memory, free - available GPU memory
    """
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', torch.cuda.current_device()))
    pynvml.nvmlInit()
    
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
            print(r'gpu_id {} is not existing!'.format(gpu_id))
            return 0, 0, 0

        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
        total = round(meminfo.total / 1024 / 1024, 2)
        used = round(meminfo.used / 1024 / 1024, 2)
        free = round(meminfo.free / 1024 / 1024, 2)
        return total, used, free

    else:
        mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
        mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
        mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
        
        return mem_total, mem_process_used, mem_free
    
class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        
        # Set the behavier of onnxruntime
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=True
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        options.intra_op_num_threads = 1
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}
        self.onnxfile = onnxfile
        self.sess = ort.InferenceSession(onnxfile, 
                                         sess_options=options, 
                                         providers=[('CUDAExecutionProvider', cuda_provider_options)])
        print(self.sess.get_providers())
        
        # model_onnx = onnx.load(onnxfile)    #load onnx    
        # print(onnx.helper.printable_graph(model_onnx.graph))    
        
        self.inputs_names = [input.name for input in self.sess.get_inputs()]
        self.output_names = [output.name for output in self.sess.get_outputs()]

        # logger.debug('{} loaded'.format(onnxfile))

    def forward(self, _inputs: dict):
       
        assert len(self.inputs_names) == len(_inputs)
        # import pdb
        # pdb.set_trace()    
        input_keys=list(_inputs.keys())
        inputs={}
        for idx,name in enumerate(self.inputs_names):
            inputs[name]=_inputs[input_keys[idx]]
    
        output_tensors = self.sess.run(None, inputs)

        assert len(output_tensors) == len(self.output_names)
        
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor
        del output_tensors
        return output
        
    def __del__(self):
        logger.debug('{} unload'.format(self.onnxfile))


@singleton
class MemoryPoolSimple:
    def __init__(self, max_memory):
        if max_memory < 0:
            raise Exception('max_gpu(cpu)_memory must > 0, get {}'.format(max_memory))
        
        self.max_size = max_memory*1024
        self.wait_map = {}
        self.active_map = {}
        self.memory_need_per_module = {} 
        self.first_infrence = True
    
    def update_memoty_need(self, key, size):
        if self.first_infrence:
            self.memory_need_per_module.update({key: size})

    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'memory_need': os.path.getsize(onnx_filepath) / 1024 / 1024, 
            }

    def find_max_memory_from_activate_map(self):
        biggest_k = None
        biggest_size = 0
        for key in self.active_map.keys():
            cur_size = self.memory_need_per_module[key]
            if biggest_k is None:
                biggest_k = key
                biggest_size = cur_size
                continue
            
            if cur_size > biggest_size:
                biggest_size = cur_size
                biggest_k = key
        
        return biggest_k

    def fetch(self, key: str):
        if key in self.active_map:
            return self.active_map[key]
        

        onnxfile = self.wait_map[key]['onnx']
        if self.first_infrence:
            return OrtWrapper(onnxfile)
        
        
        # check current memory use
        biggest_key = self.find_max_memory_from_activate_map()
        need = self.memory_need_per_module[key]
        total, used, free = get_mem_info()
        
        while biggest_key is not None and need > free:
            # if exceeded once, delete until `max(half_max, file_size)` left
            if len(self.active_map) == 0:
                break
           
            del self.active_map[biggest_key]
            biggest_key = self.find_max_memory_from_activate_map()
            total, used, free = get_mem_info()
        
        self.active_map[key] = OrtWrapper(onnxfile)
        return self.active_map[key]
