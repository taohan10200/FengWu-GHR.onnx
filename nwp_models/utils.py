import numpy as np
from threading import Lock
import psutil
import os
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

def singleton(cls):
    _instance = {}
    _instance_lock = Lock()

    def inner(*args, **kwargs):
        if cls not in _instance:
            with _instance_lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


def npsoftmax(x, axis):
    y = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)


def npmultinominal2D(x):
    assert len(x.shape) == 2

    ret = np.zeros((x.shape[0], 1), dtype=x.dtype)

    for row, pval in enumerate(x):
        ret[row] = np.random.multinomial(1, pval).argmax()

    return ret


if __name__ == '__main__':
    data = np.ones((12, 8))
    data1 = npsoftmax(data, -1)

    data2 = npmultinominal2D(data1)
    print(data2)
