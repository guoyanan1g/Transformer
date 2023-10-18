from transformer_input import *
from encoder import *
from decoder import *
from generator import *

dim=512
head=8
vocalb_size=5000

class EncoderAndDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super(EncoderAndDecoder,self).__init__()

        self.encoder=encoder
        self.encoder=decoder
        self.src_embed=source_embed
        self.trg_embed=target_embed
        self.generator=generator

    def encode(self,source,source_mask):
        return self.encoder(source,source_mask)
    def decode(self,memory,source_mask,target,target_mask):
        return self.decoder(self.trg_embed(target),memory,source_mask,target_mask)

    def forward(self,source,target,source_mask,target_mask):
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)

cpy=copy.deepcopy
class Transformer():
    def __init__(self,source_vocalb_size,target_vocalb_size,N=6,dim=512,dim_for_feed_forward=2048,head=8,dropout=0.1):
        ff = FeedForwardLayer(dim, dim_for_feed_forward)
        attn = MutiHeadAttention(dim, head, 0.2)
        pos = PositionEncoding(dim)
        embed = Embeddings(vocalb_size, dim)
        ge = Generator(dim, vocalb_size)
        self.model = EncoderAndDecoder(
            Encoder(EncoderLayer(cpy(attn), cpy(ff), dim), N),
            Decoder(DecoderLayer(cpy(attn), cpy(attn), cpy(ff), dim, dropout), N),
            nn.Sequential(embed, cpy(pos)),
            nn.Sequential(embed, cpy(pos)),
            ge
        )

        for p in self.model.parameters():
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p) #均匀分布

def main():
    t=Transformer(vocalb_size,vocalb_size)
    print(t.model)

if __name__ == '__main__':
    main()