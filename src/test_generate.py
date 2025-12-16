import time
import torch
import torch.nn.functional as F
from src.tokenizer import tokenizer
from src.config import config
from src.gpt import GPT
from tqdm import tqdm

@torch.inference_mode()
def generate(model,device,tokenizer,seed_txt,max_len = 200):
    temp = 0.6
    model.eval()
    start_pos = 0
    seed_tokens = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(seed_txt),device = device).unsqueeze(0)
    attn_mask = torch.ones_like(seed_tokens[:,:-1]).to(device)
    model(seed_tokens[:,:-1],attn_mask,start_pos,use_rope = True)
    start_pos = seed_tokens.size(1) - 1  # Sequence length of processed tokens
    offset = torch.tensor([[1]],dtype = torch.long,device=device)
    predicted_token = seed_tokens[:, -1:]
    generated = []
    for _ in tqdm(range(max_len)):
        # Extend mask to cover the position where new token will be stored
        attn_mask = torch.cat([attn_mask, offset], dim=1)
        logits = model(predicted_token, attn_mask, start_pos, use_rope=True)
        if len(logits.shape) == 4:
            next_token_logits = logits[0, 0, -1, :]  
        elif len(logits.shape) == 3:
            next_token_logits = logits[0, -1, :]  
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        probs = F.softmax(next_token_logits / temp, dim=-1)
        vocab_size = len(tokenizer)
        idx = torch.multinomial(probs, num_samples=1)
        idx_scalar = torch.clamp(idx, 0, vocab_size - 1).item()
        generated.append(idx_scalar)
        predicted_token = idx.unsqueeze(0)
        start_pos += 1
        if idx_scalar == tokenizer.eos_token_id:
            break
        
    return tokenizer.decode(generated)  
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_txt = "once upon a time"
    model = GPT(config.n_embd,config.n_head,config.n_layer,config.max_seq_len,tokenizer.vocab_size)
    model.load_state_dict(torch.load("assets/3hr_gpt_model.pth", map_location=device), strict=False)
    model.to(device)
    st = time.time()
    txt = generate(model,device,tokenizer,seed_txt)
    end = time.time()
    print(end-st)
    print(txt)