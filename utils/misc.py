import os
import time
import numpy as np
import torch

def model_save(model, new_file):
    print("Saving model to {}".format(new_file))
    torch.save(model, new_file)

