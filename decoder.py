import nntplib

from  transformer_attention import *
from encoder import SubLayer,NormLayer


class DecoderLayer(nn.Module):
    def __init__(self,muti_head_attention,muti_head_self_attention,feed_forward,dim,dropout):
        super(DecoderLayer,self).__init__()
        self.mha=muti_head_attention
        self.self_mha=muti_head_self_attention
        self.ff=feed_forward
        self.dim=dim

        self.sublayer=nn_clone(SubLayer(dim,dropout),3)
    def forward(self,x,memory,source_mask,target_mask): #memory是编码器encoder的输出

        x=self.sublayer[0](x,lambda x:self.mha(x,x,x,target_mask))
        x=self.sublayer[1](x,lambda x:self.self_mha(x,memory,memory,source_mask))
        return self.sublayer[2](x,self.ff)


class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=nn_clone(layer,N)
        self.norm=NormLayer(layer.dim)


    def forward(self,x,memory,source_mask,target_mask):
        for layer in self.layers:
            x=layer(x,memory,source_mask,target_mask)
        return self.norm(x)