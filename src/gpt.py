import torch
from torch import nn
from src.config import config
from src.attention import MultiHeadAttentionVec
from src.rope import PE_Vec,RMS_Norm
from src.moe import MoE,FeedForwardNet
from src.tokenizer import tokenizer

class TransformerDecoderBLK(nn.Module):
    def __init__(self,d_model,heads,use_rope = False):
        super(TransformerDecoderBLK,self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.use_rope = use_rope
        self.attention = MultiHeadAttentionVec(d_model,heads,mask = True)
        
        if not use_rope:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = FeedForwardNet(d_model)
        else:
            self.norm1 = RMS_Norm(d_model)
            self.norm2 = RMS_Norm(d_model)
            self.ffn = MoE(d_model)

    def forward(self,x,att_mask,start_pos,use_rope = False): 
        x = x + self.attention(self.norm1(x),att_mask,start_pos,use_rope)
        x = x + self.ffn(self.norm2(x))
        return x
        
class GPT(nn.Module):
    def __init__(self,d_model,heads,depth,max_len,vocab_size):
        super(GPT,self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.max_len = max_len
        self.embeddings = nn.Embedding(vocab_size,d_model)

        self.layers = nn.ModuleList([TransformerDecoderBLK(d_model,heads) for _ in range(depth)])
        self.pos_encoding = PE_Vec(d_model,max_len)
        self.unembedding = nn.Linear(d_model,vocab_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self,x,att_mask,start_pos,use_rope = False):
        batch_size,seq_len = x.shape
        x = self.dropout(self.embeddings(x))
        if not use_rope:
            x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x,att_mask,start_pos,use_rope)
        return self.unembedding(x)  
    
if __name__ == "__main__":
    model = GPT(config.n_embd,config.n_head,config.n_layer,config.max_seq_len,tokenizer.vocab_size)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)