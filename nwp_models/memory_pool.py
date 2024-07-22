# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
from loguru import logger
from .utils import singleton, get_mem_info
import onnxruntime as ort
import numpy as np
import os
import sys
import psutil
import onnx
import math
class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        
        # Set the behavier of onnxruntime
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=False
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
        # print(onnx.helper.printable_graph(model_onnx.graph))    #
        
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
    def __init__(self, poolsize_GB):
        if poolsize_GB < 0:
            raise Exception('poolsize_GB must > 0, get {}'.format(poolsize_GB))
        
        self.max_size = poolsize_GB*1024
        self.wait_map = {}
        self.active_map = {}
        
    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'memory_need': os.path.getsize(onnx_filepath) / 1024 / 1024, 
            }

    def used(self):
        sum_size = 0
        biggest_k = None
        biggest_size = 0
        for key in self.active_map.keys():
            cur_size = self.wait_map[key]['memory_need']
            sum_size += cur_size

            if biggest_k is None:
                biggest_k = key
                biggest_size = cur_size
                continue
            
            if cur_size > biggest_size:
                biggest_size = cur_size
                biggest_k = key
        
        return sum_size, biggest_k

    def check(self):
        sum_need = 0
        for k in self.wait_map.keys():
            sum_need = sum_need + self.wait_map[k]['memory_need']
            
        sum_need /= (1024 * 1024 * 1024)
        
        total = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        # import pdb
        # pdb.set_trace()
        if total > 0 and total < sum_need:
            logger.warning('virtual_memory not enough, require {}, try `--poolsize {}`'.format(sum_need, math.floor(total)))


    def fetch(self, key: str):
        if key in self.active_map:
            return self.active_map[key]
        

        onnxfile = self.wait_map[key]['onnx']
        
        # check current memory use
        active_used_size, biggest_key = self.used()
        need = max(self.max_size / 2, self.wait_map[key]['memory_need'])
        total, used, free = get_mem_info()
        
        while biggest_key is not None and need > free:
            # if exceeded once, delete until `max(half_max, file_size)` left
            need = max(need, self.max_size * 3/4)
            if len(self.active_map) == 0:
                break
           
            del self.active_map[biggest_key]
            active_used_size, biggest_key = self.used()
            total, used, free = get_mem_info()
        
        self.active_map[key] = OrtWrapper(onnxfile)
        return self.active_map[key]
