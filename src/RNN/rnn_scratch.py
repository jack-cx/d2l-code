from os import stat
import torch
from torch import nn
from d2l import torch as d2l

# define RNN model
class RNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h  = nn.Parameter(torch.zeros(num_hiddens))

@d2l.add_to_class(RNNScratch)
def forward(self, inputs, state=None):
    if state is not None:
        
        state, = state