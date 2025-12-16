from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpt import GPT
from src.config import config
from src.tokenizer import tokenizer
from src.dataloader import train_loader,val_loader
from src.dataloader import train_loader,val_loader
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast,GradScaler
import time
import matplotlib.pyplot as plt
import os
import warnings
import wandb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate(model,val_loader,criterion,device):
    model.eval()
    avg_val_losses = 0
    for x in tqdm(val_loader):
        with torch.no_grad():
            input_ids = x["input_ids"].to(device)
            attn_mask = x["attention_mask"].to(device)
    
            inputs = input_ids[:,:-1]
            targets = input_ids[:,1:]

            with torch.amp.autocast(device_type = config.device,dtype = torch.float16):
                logits = model(inputs,attn_mask[:,:-1],use_rope = True)
                logits_view = logits.contiguous().view(-1,config.vocab_size)
                targets_view = targets.contiguous().view(-1)
                loss = criterion(logits_view,targets_view)

            avg_val_losses += loss.item()
    
    avg_val_loss = avg_val_losses / len(val_loader)
    wandb.log({"val_loss": avg_val_loss})
    return avg_val_loss


def train():
    losses = []
    best_val_loss = float('inf')
    patience = 5  
    patience_counter = 0
    scaler = GradScaler()  

    # Track metrics
    train_losses = []
    val_losses = []
    training_start_time = time.time()

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for x in progress_bar:
            optimizer.zero_grad()
            input_ids = x["input_ids"].to(config.device)
            attn_mask = x["attention_mask"].to(config.device)

            inputs = input_ids[:,:-1]
            targets = input_ids[:,1:]
            with autocast(device_type = config.device,dtype = torch.float16):
                logits,_ = model(inputs,attn_mask[:,:-1],use_rope = True)
                logits_view = logits.contiguous().view(-1,config.vocab_size)
                targets_view = targets.contiguous().view(-1)
                loss = criterion(logits_view,targets_view)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = loss.item()
            losses.append(loss_value)
            epoch_losses.append(loss_value)

            # Log to wandb every step
            wandb.log({
                "train_loss": loss_value,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm.item(),
                "epoch": epoch + 1
            })

            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                "grad_norm": f"{grad_norm.item():.4f}"
            })

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_epoch_loss)

        # Log epoch-level metrics
        wandb.log({
            "epoch_train_loss": avg_epoch_loss,
            "epoch": epoch + 1
        })

        val_loss = validate(model,val_loader,criterion,config.device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint for this epoch
        checkpoint_path = f"transformer_checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_epoch_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Log checkpoint as wandb artifact
        artifact = wandb.Artifact(
            name=f"gpt-26m-checkpoint-epoch-{epoch}",
            type="model",
            metadata={
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            }
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"Checkpoint saved and logged to wandb: {checkpoint_path}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = "best_transformer_model.pth"
            torch.save(model.state_dict(), best_model_path)
            
            # Log best model as artifact
            best_artifact = wandb.Artifact(
                name="gpt-26m-best-model",
                type="model",
                metadata={
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_loss": avg_epoch_loss
                }
            )
            best_artifact.add_file(best_model_path)
            wandb.log_artifact(best_artifact)
            
            print(f"New best model saved with validation loss: {val_loss:.4f}")
            wandb.log({"best_val_loss": best_val_loss})
        else:
            patience_counter += 1
            wandb.log({"patience_counter": patience_counter})
            if patience_counter >= patience:
                print("Early stopping triggered")
                wandb.log({"early_stopped": True})
                break

        # Print Epoch Summary
        print(f"  Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Time elapsed: {(time.time() - training_start_time)/60:.2f} minutes")
        print("-" * 50)
        
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    wandb.log({"training_history": wandb.Image('training_history.png')})
    plt.show()

    training_time = (time.time() - training_start_time)/60
    print(f"Training completed in {training_time:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Log final metrics
    wandb.log({
        "total_training_time_minutes": training_time,
        "final_best_val_loss": best_val_loss
    })
    
    wandb.finish()    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    max_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(config.n_embd,config.n_head,config.n_layer,config.max_seq_len,tokenizer.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(),lr = config.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    num_warmup_steps = 5000  
    num_training_steps = len(train_loader) * max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    wandb_run = wandb.init(
        entity = "akshithmarepally-akai",
        project = "gpt_26m",
        config = {
            "architecture" : "GPT",
            "dataset" : "roneneldan/TinyStories",
            "configs" : config,
        }
    )
    print(f"Total Trainable Parameters : {count_parameters(model)}")
