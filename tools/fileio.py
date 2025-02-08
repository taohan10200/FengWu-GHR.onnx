import io
import torch
import numpy as np
from PIL import Image

class BaseLoader(object):
    def __init__(self):
        self._client = None
        try:
            from petrel_client import client
            self._client = client.Client()
        except:
            pass

    def load(self, path):
        if 's3://' in path:
            if self._client is not None:
                return self._load_ceph(path)
            else:
                raise RuntimeError('Please install petrel_client to enable PetrelBackend.')
        else:
            return self._load_local(path)

    def save(self, path, data):
        if 's3://' in path:
            if self._client is not None:
                self._save_ceph(path, data)
            else:
                raise RuntimeError('Please install petrel_client to enable PetrelBackend.')
        else:
            self._save_local(path, data)

    def _load_ceph(self, path):
        raise NotImplementedError()

    def _load_local(self, path):
        raise NotImplementedError()

    def _save_ceph(self, path, data):
        raise NotImplementedError()

    def _save_local(self, path, data):
        raise NotImplementedError()


class NumpyLoader(BaseLoader):
    def _load_ceph(self, path):
        file_bytes = self._client.get(path)
        file_buff = io.BytesIO(file_bytes)
        data = np.load(file_buff)
        return data

    def _load_local(self, path):
        return np.load(path)

    def _save_ceph(self, path, data):
        with io.BytesIO() as f:
            np.save(f, data)
            f.seek(0)
            self._client.put(path, f)

    def _save_local(self, path, data):
        np.save(path, data)


class TorchLoader(BaseLoader):
    def _load_ceph(self, path):
        file_bytes = self._client.get(path)
        file_buff = io.BytesIO(file_bytes)
        data = torch.load(file_buff, map_location='cpu')
        return data

    def _load_local(self, path):
        return torch.load(path, map_location='cpu')

    def _save_ceph(self, path, data):
        with io.BytesIO() as f:
            torch.save(data, f)
            f.seek(0)
            self._client.put(path, f)

    def _save_local(self, path, data):
        torch.save(data, path)


class PILLoader(BaseLoader):
    def _load_ceph(self, path):
        file_bytes = self._client.get(path)
        file_buff = io.BytesIO(file_bytes)
                
        img = Image.open(file_buff)
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        img = img.convert('RGB')
        return img

    def _load_local(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert('RGB')
