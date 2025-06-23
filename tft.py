
import torch.nn as nn

class TFTChurn(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.linear(x)
