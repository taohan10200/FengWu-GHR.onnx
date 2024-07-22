# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import onnxruntime as ort
import numpy as np
import os
from loguru import logger
from .memory_pool import MemoryPoolSimple, get_mem_info

class FengWu_GHR:
    
    def __init__(self, pool: MemoryPoolSimple, onnxdir: str, onnx_keys:list):
        assert os.path.isdir(onnxdir)
        self.memory_pool = pool
        self.onnx_keys = onnx_keys
        
        for key in onnx_keys:
            filepath = os.path.join(onnxdir, f'{key}.onnx')
            self.memory_pool.submit(key, filepath)

    def one_step(self, inputs={'input':np.array, 
                               'step': np.array}):
        import time 
       
        x = inputs
        st=time.time()
        for key in  self.onnx_keys:

            handler = self.memory_pool.fetch(key)
            
            total, used_before, free = get_mem_info()
            x = handler.forward(x)
            # print(x['output'])
            total, used_after, free = get_mem_info()
            self.memory_pool.update_memoty_need(key, (used_after-used_before))
    
        print(f'one step time is {time.time()-st}')

        self.memory_pool.first_infrence=False
        return x
