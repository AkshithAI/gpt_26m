import torch
from src.tokenizer import tokenizer

class GPTConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = tokenizer.vocab_size
        self.n_layer = 6          
        self.n_head = 8           
        self.n_embd = 512         
        self.max_seq_len = 512    
        self.no_of_experts = 6
        self.num_experts_per_tok = 2
        # Training
        self.dropout_rate = 0.1
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

config = GPTConfig()