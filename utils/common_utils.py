import json
import torch.nn as nn
from easydict import EasyDict as edict

def process_config(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = edict(config_dict)
    return config

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x

def try_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x