import torch
import torch.nn as nn
from torch.autograd import Variable
from math import sqrt
import torch.nn.functional as F
import copy

#attention:注意力机制
#1.mat-multipe(Q,K)  2.Scale  3.Mask  4.SoftMax  5.mat-multiple with V
#SoftMax(Q*K_T/sqrt(dim))*V  dim is word_embedding_dim

def attention(query,key,value,mask=None,dropout=None):
    dim=query.size(-1)
    t=torch.matmul(query,key.transpose(-2,-1))/sqrt(dim)
    if mask is not None:
        t=t.masked_fill(mask==0,1e-9)
    attention_tensor=F.softmax(t,dim=-1)
    if dropout is not None:
        attention_tensor=dropout(attention_tensor)
    return torch.matmul(attention_tensor,value),attention_tensor


# query=key=value=torch.rand(2,4,512)
# attn,attn_t=attention(query,key,value)
# print(attn,attn_t,attn.shape,attn_t.shape)

def nn_clone(Module,N): #deep copy a layer for N times
    return nn.ModuleList([copy.deepcopy(Module) for i in range(N)])


""""
        Linear
          ^
          |
        Concat
          ^
          |
    Scaled Dot-Product attention
          ^
          |
        Linear
          ^
          |
       (Q,K,V)
"""
class MutiHeadAttention(nn.Module): #多头注意力
    def __init__(self,dim,head,dropout=0.1):
        super(MutiHeadAttention,self).__init__()
        assert dim%head==0

        self.dim_per_head=dim//head #词嵌入维度划分为多个头
        self.head=head
        self.attention_tensor=None
        self.dropout=nn.Dropout(dropout)
        self.linears=nn_clone(nn.Linear(dim,dim),4)

    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)
        if mask is not None:
            mask=mask.unsqueeze(1) #扩充至3-dim,该维度代表head数

        #pytorch的view是从后往前的，所以先确保维数是准确的,-1的维数代表句子长度
        query,key,value=[model(x).view(batch_size,-1,self.head,self.dim_per_head).transpose(1,2) for model,x in zip(self.linears,(query,key,value))]
        x,self.attention_tensor=attention(query,key,value,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.dim_per_head)#最后一维还原为embedding_dim

        return self.linears[-1](x)

# query=key=value=torch.rand(2,4,512)
# mask=torch.zeros(2,4,4)
# model=MutiHeadAttention(512,8)
# res=model(query,key,value,mask)
# print(res.shape)




