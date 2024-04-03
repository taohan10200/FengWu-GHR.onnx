# Copyright (c) Tao Han: hantao10200@gmail.com. All rights reserved.
import onnxruntime as ort
import numpy as np
import os
from loguru import logger
from .memory_pool import MemoryPoolSimple
from .utils import get_mem_info

class FengWu_GHR:
    
    def __init__(self, pool: MemoryPoolSimple, onnxdir: str, onnx_keys:list):
        assert os.path.isdir(onnxdir)
        self._pool = pool
        self.onnx_keys = onnx_keys
        
        for key in onnx_keys:
            filepath = os.path.join(onnxdir, f'{key}.onnx')
            self._pool.submit(key, filepath)
        self.past_allocated_memory = 0         
        
    def one_step(self, inputs={'input':np.array, 
                               'step': np.array}):
        import time 
       
        x = inputs
        st=time.time()
        for key in  self.onnx_keys:

            handler = self._pool.fetch(key)
            x = handler.forward(x)
            # print(x['output'])
            total, used, free = get_mem_info()
            self._pool.wait_map[key]['memory_need'] = used-self.past_allocated_memory
            self.past_allocated_memory = used
            
        print(f'one step time is {time.time()-st}')
            # if 'block_3' in key:
            #     import pdb
            #     pdb.set_trace()  
        return x
