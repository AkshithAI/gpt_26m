from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
tokenizer = Tokenizer.from_file("/Users/apple/Documents/API/assets/tinystories_tokenizer1.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer)
tokenizer.pad_token_id = 0
tokenizer.unk_token_id = 1
tokenizer.bos_token_id = 2
tokenizer.eos_token_id = 3