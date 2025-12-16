from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from .tokenizer import tokenizer
from .config import config

# Load TinyStories
dataset = load_dataset("roneneldan/TinyStories")
train_data = dataset["train"]
val_data = dataset["validation"]

class TinyStoryDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        encoded = tokenizer(self.data[idx],
                    padding = "max_length",
                    truncation = True,
                    max_length = config.max_seq_len,
                    return_tensors = "pt",
                    return_attention_mask = True 
                   )
        return {
            "input_ids" : encoded["input_ids"].squeeze(0),
            "attention_mask" : encoded["attention_mask"].squeeze(0)
        }

train_data = TinyStoryDataset(train_data["text"])
val_data = TinyStoryDataset(val_data["text"])

train_loader = DataLoader(train_data,batch_size = 64,shuffle = True,pin_memory = True)
val_loader = DataLoader(val_data,batch_size = 64,shuffle = False,pin_memory = True)



if __name__ == "__main__":
    print(len(train_loader))
    print(len(val_loader))