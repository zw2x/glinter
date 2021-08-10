import logging
import traceback
from collections import OrderedDict
from pathlib import Path
import shutil

import torch
from torch.serialization import default_restore_location


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_state(path, states):
    torch_persistent_save(states, path)

def load_state(path,):
    path = Path(path)
    if not path.exists():
        return
    state = torch.load(path, map_location=lambda s, l: default_restore_location(s, 'cpu'))

    return state

if __name__ == '__main__':
    import sys
    state = load_state(sys.argv[1])
    save_state(sys.argv[2], state['model'])
