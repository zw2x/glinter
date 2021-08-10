# modified by zw2x from torch_geometric/data/dataloader.py

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch

class DefaultCollater(object):
    def __init__(self, strict=False):
        self.strict = strict 

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        if self.strict:
            raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class Collater:
    def __init__(self):
        self.collate_fn = DefaultCollater()

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, dict):
            sample = {}
            for elem in batch:
                for k in elem:
                    if k not in sample:
                        sample[k] = []
                    sample[k].append(elem[k])
            for k, b in sample.items():
                _sample = self(b)
                if _sample is not None:
                    sample[k] = _sample
        elif isinstance(elem, Batch):
            if len(batch) == 1:
                return elem
            else:
                raise NotImplementedError
        else:
            sample = self.collate_fn(batch)

        return sample
