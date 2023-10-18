import torch
import torch.nn as nn
from math import sqrt,log
from torch.autograd import Variable

#输入层由词嵌入层和位置编码器组成。
class Embeddings(nn.Module):
    def __init__(self,vocalb_size,dim):
        super(Embeddings,self).__init__()
        self.embed=nn.Embedding(vocalb_size,dim)
        self.dim=dim

    def forward(self,x):
        return self.embed(x)*sqrt(self.dim)

class PositionEncoding(nn.Module):
    def __init__(self,dim,dropout=0.3,max_len=100): #max_len表示句子的最大长度
        super(PositionEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)

        position_encoding_matrix=torch.zeros(max_len,dim)
        p=torch.arange(0,max_len).unsqueeze(1) #size=[max_len,1]
        q=torch.exp(torch.arange(0,dim,2)*(-log(10000)/dim)).unsqueeze(0) #size=[1,dim]
        #奇数列为sin，偶数列为cos，三角函数有很好的缩放效果
        position_encoding_matrix[:,0::2]=torch.sin(p*q) #size(p*q)=[max_len,dim]
        position_encoding_matrix[:,1::2]=torch.cos(p*q)
        position_encoding_matrix=position_encoding_matrix.unsqueeze(0) #为了与x保持size一致，扩充一个batch维

        self.register_buffer('pem',position_encoding_matrix)#该矩阵无需参与优化,buffer不属于超参，不参与更新
    def forward(self,x): #x：[N,S,D] batch_first=True
        x=x+Variable(self.pem[:,:x.size(1),:],requires_grad=False) #第二维是max_len，截取句子的长度和x一致。等价于[:,:x.size(1)]
        return self.dropout(x)


if __name__=='__main__':
    x=Variable(torch.LongTensor([[1,2,3,4],[5,6,7,8]])) #[2,4,1]
    emd=Embeddings(1000,512)
    x=emd(x)
    p=PositionEncoding(512)
    x=p(x)
    print(x.shape)


