
import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()

class MinMaxObserver(ObserverBase):
    def __init__(self):
        super(MinMaxObserver, self).__init__()
        self.num_flag = 0

        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

    @torch.no_grad()
    def forward(self, input):
        min_val = torch.min(input)
        max_val = torch.max(input)

        self.update_range(min_val, max_val)

        return input

