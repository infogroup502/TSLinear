import torch
import math
from torch import nn

class w_cheng(nn.Module):
    def __init__(self, c, dime):
        super(w_cheng,self).__init__()
        self.device = 'cuda'
        self.c=c
        self.dime=dime
        self.w_m = torch.nn.Parameter(torch.ones(self.c, self.dime).to(self.device).requires_grad_())
        # nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
    def forward(self, x):
        x = x * self.w_m
        return x

class w_jia(nn.Module):
    def __init__(self, c, dime):
        super(w_jia,self).__init__()
        self.device = 'cuda'
        self.c=c
        self.dime=dime
        self.w_m = torch.nn.Parameter(torch.ones(self.c, self.dime).to(self.device).requires_grad_())
    def forward(self, x):
        x = x + self.w_m
        return x

class w_ju(nn.Module):
    def __init__(self, c, dime):
        super(w_ju,self).__init__()
        self.device = 'cuda'
        self.c=c
        self.dime=dime
        self.w_m = torch.nn.Parameter(torch.randn(self.c, self.dime).to(self.device).requires_grad_())
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
    def forward(self, x):
        x = torch.matmul(x,self.w_m)
        return x

