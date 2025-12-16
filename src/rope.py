from torch import nn
import torch
import torch.nn.functional as F
from src.config import config
import math

class RoPE(nn.Module):
    def __init__(self,rope_theta,d_model):
        super().__init__()
        self.rope_theta = rope_theta
        self.dim = d_model
        self.freq_cis = self.precompute_freq_cis(config.max_seq_len,config.device)
    def precompute_freq_cis(self,end,device):
        freqs = 1.0 / self.rope_theta ** (torch.arange(0,self.dim,2)[:(self.dim//2)].float() / self.dim)
        t = torch.arange(end,device = freqs.device)
        freqs = torch.outer(t ,freqs) #(seq_len,d_model//2)
        return torch.polar(torch.ones_like(freqs),freqs).to(device) 

    def apply_rotary_emb(self,end,xq,xk,offset : int):
        batch_size, heads, seq_len, d_k = xq.shape
        freq_cis = self.freq_cis[offset:offset+seq_len,:]
        #print(batch_size, heads, seq_len, d_k)
        #print(xq.shape,xk.shape)
        xq_ = torch.view_as_complex(xq.float().reshape(batch_size, heads, seq_len,-1,2)) # (B,S,d_model//2,2)
        xk_ = torch.view_as_complex(xk.float().reshape(batch_size, heads, seq_len,-1,2)) # (B,S,d_model//2,2)
        freq_cis = freq_cis.unsqueeze(0).unsqueeze(0)
        # print(xq_.shape,freq_cis.shape)
        xq_out = torch.view_as_real(xq_ * freq_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freq_cis).flatten(3)
        return xq_out.type_as(xq),xk_out.type_as(xk)
    


class PE_Vec(nn.Module):
    def __init__(self,d_model,max_len = 512):
        super(PE_Vec,self).__init__()
        pos = torch.arange(0,max_len,dtype = torch.float).unsqueeze(1)
        pe = torch.zeros(max_len,d_model)
        div_term = torch.exp(torch.arange(0,d_model,2) * - math.log(10000)/d_model)
        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self,x):
        return self.add_with_offset(x,0)

    def add_with_offset(self,x,offset):
        batch_size,seq_len,_ = x.shape
        positions = torch.arange(offset, offset + seq_len, device=x.device).unsqueeze(0)
        pos_enc = self.pe[:,positions,:].to(x.device)
        return x + pos_enc
    

class RMS_Norm(nn.Module):
    def __init__(self,num_features,eps : float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self,x):
        t,dtype = x.float(),x.dtype
        t = t * torch.rsqrt(torch.mean(t**2,dim = -1,keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)