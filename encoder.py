from torch.nn.functional import relu
import torch.nn as nn
import torch
from transformer_attention import  nn_clone


class FeedForwardLayer(nn.Module): #前馈全连接层
    def __init__(self,embedding_dim,mid_dim,dropout=0.1):
        super(FeedForwardLayer,self).__init__()
        self.l1=nn.Linear(embedding_dim,mid_dim)
        self.l2=nn.Linear(mid_dim,embedding_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.l2(self.dropout(relu(self.l1(x))))

class NormLayer(nn.Module): #规范层
    def __init__(self,dim):
        super(NormLayer,self).__init__()
        self.n1=nn.Parameter(torch.ones(dim))
        self.n2=nn.Parameter(torch.zeros(dim))
        self.eps=1e-6
    def forward(self,x):
        mean=x.mean(-1,keep_dim=True)
        std=x.std(-1,keep_dim=True)
        return self.n1*(x-mean)/(std+self.eps)+self.n2

class SubLayer(nn.Module): #residual 子层，就是一个残差连接
    def __init__(self,dim,dropout=0.1):
        super(SubLayer,self).__init__()
        self.norm=NormLayer(dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self,muti_head_attention,feed_forward,dim):
        super(EncoderLayer,self).__init__()
        self.dim=dim
        self.mha=muti_head_attention
        self.ff=feed_forward
        self.sublayer=nn_clone(SubLayer(dim),2)

    def forward(self,x,mask):
        output=self.sublayer[0](x,lambda x:self.mha(x,x,x,mask))
        return self.sublayer[1](output,self.ff)

class Encoder(nn.Module):
    def __init__(self,encoder_layer,N):
        super(Encoder,self).__init__()
        self.el=nn_clone(encoder_layer,N)
        self.norm=NormLayer(encoder_layer.dim)


    def forward(self,x,mask):
        for layer in self.el:
            x=layer(x,mask)
        return self.norm(x)
