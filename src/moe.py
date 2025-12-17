from torch import nn
import torch
import torch.nn.functional as F
from src.config import config

class FeedForwardNet(nn.Module):
    def __init__(self,d_model,dropout = 0.2):
        super().__init__()
        self.d_model = d_model
        self.w_1 = nn.Linear(d_model,4 * d_model)
        self.w_2 = nn.Linear(4 * d_model , d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.w_1(x)
        x = F.silu(x)
        x = self.dropout(x)
        final = self.w_2(x)
        return final
    
class MoE(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.gate = nn.Linear(d_model,config.no_of_experts,bias = False)
        self.experts = nn.ModuleList([FeedForwardNet(d_model,config.dropout_rate) for _ in range(config.no_of_experts)])
        
    def forward(self,x):
        gate_logits = self.gate(x) 
        xprt_weights,xprt_idxs = torch.topk(gate_logits,config.num_experts_per_tok,dim = -1)
        # xprt_weights = xprt_weights / xprt_weights.sum(dim = -1 ,keepdim = True)
        xprt_weights = F.softmax(xprt_weights,dim = -1)
        output = torch.zeros_like(x)
        for i,expert in enumerate(self.experts):
            batch_idx,seq_idx,expert_idx = torch.where(xprt_idxs == i)
            output[batch_idx,seq_idx] += xprt_weights[batch_idx,seq_idx,expert_idx,None] * expert(x[batch_idx,seq_idx])
        return output