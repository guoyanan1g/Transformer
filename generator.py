import torch.nn as nn
from torch.nn.functional import softmax

#输出部分的类别生成器
class Generator(nn.Module):
    def __init__(self,dim,vocalb_size):
        super(Generator,self).__init__()
        self.linear=nn.Linear(dim,vocalb_size)
    def forward(self,x):
        return softmax(self.linear(x),dim=-1)