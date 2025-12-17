import torch
from torch import nn
import torch.nn.functional as F
from src.config import config
import math
from src.rope import RoPE

class MultiHeadAttentionVec(nn.Module):
    def __init__(self,d_model,heads = 8,mask = True):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = self.d_model // self.heads
        self.mask = mask
        self.rope = RoPE(10000,self.d_k)
        self.w_q = nn.Linear(d_model,d_model,bias = False)
        self.w_k = nn.Linear(d_model,d_model,bias = False)
        self.w_v = nn.Linear(d_model,d_model,bias = False)
        self.register_buffer("cache_k",torch.zeros(1,heads,config.max_seq_len,self.d_k))
        self.register_buffer("cache_v",torch.zeros(1,heads,config.max_seq_len,self.d_k))
        self.w_o= nn.Linear(d_model,d_model,bias = False)

    def clear_cache(self):
        """Clear the KV cache - call before starting a new generation"""
        self.cache_k.zero_()
        self.cache_v.zero_()

    def forward(self,x,att_mask,start_pos=0,use_rope=False,use_cache=False): 
        if len(x.shape) == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        batch_size,seq_len,_ = x.shape
        
        Q = self.w_q(x).view(batch_size,seq_len,self.heads,self.d_k).transpose(1,2)
        K = self.w_k(x).view(batch_size,seq_len,self.heads,self.d_k).transpose(1,2)
        V = self.w_v(x).view(batch_size,seq_len,self.heads,self.d_k).transpose(1,2)
        
        if use_rope:
            Q,K = self.rope.apply_rotary_emb(seq_len,Q,K,offset=start_pos if use_cache else 0)
        
        if use_cache:
            # KV caching for inference
            end_pos = start_pos + seq_len
            self.cache_k[:,:,start_pos:end_pos,:] = K
            self.cache_v[:,:,start_pos:end_pos,:] = V
            K = self.cache_k[:,:,:end_pos,:]
            V = self.cache_v[:,:,:end_pos,:]
        
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k) #(B,H,S,S)
        
        if att_mask is not None:
            pad_mask = att_mask.unsqueeze(1).unsqueeze(1)  # Shape: [B, 1, 1, full_seq_len]
            scores = scores.masked_fill(pad_mask == 0, -1e9)
        
        # Apply causal mask during training or prefill (seq_len > 1)
        if self.mask and seq_len > 1:
            full_seq_len = scores.shape[-1]
            causal_mask = torch.tril(torch.ones(seq_len,full_seq_len,dtype=torch.bool,device=x.device))
            scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        att_scores = F.softmax(scores,dim=-1)
        final = torch.matmul(att_scores,V).transpose(1,2).contiguous().view(batch_size,seq_len,self.heads * self.d_k)
        out = self.w_o(final)
        return out
