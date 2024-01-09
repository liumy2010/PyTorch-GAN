import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CopyPaste(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []
        self.stats = []
        self.num = 0
        self.grad_norm_sum = 0

    def Get_Distance(self):
        loss = 0
        for idx, p in enumerate(self.parameters()):
            loss += (torch.norm(p - self.data[idx]))**2
        return loss

    @torch.no_grad()
    def copy(self):
        self.data = []
        self.stats = []
        for idx, p in enumerate(self.parameters()):
            self.data.append(p.data.clone().detach())
            self.stats.append(p.data.sum().item())
    
    @torch.no_grad()
    def paste(self):
        for idx, p in enumerate(self.parameters()):
            #print(self.stats[idx],self.data[idx].sum().item() )
            #assert(np.abs(self.stats[idx] - self.data[idx].sum().item())<1e-9)
            self.grad_norm_sum += torch.norm(p.grad.data).item()
            p.data.copy_(self.data[idx])
        self.num += 1
        if self.num % 100==0:
            print(self.grad_norm_sum / self.num)
    
    @torch.no_grad()
    def add_sling(self, lr, sling):
        for idx, p in enumerate(self.parameters()):
            p.data.add_(lr * sling * self.data[idx])
    
    @torch.no_grad()
    def assert_check(self):
        for idx, p in enumerate(self.parameters()):
            assert(np.abs(self.stats[idx] - p.data.sum().item())<1e-9)