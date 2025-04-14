# lane2seq_project/train.py

import os
import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from datasets.lane_dataset import TuSimpleDataset
from torch.utils.data import DataLoader
from utils.collate_fn import collate_fn  
from models.lane2seq import Lane2Seq
from utils.tokenizer import LaneTokenizer

# from datasets.lane_dataset import LaneDataset  # We'll write this soon
# from utils.logger import setup_logger  # Optional for future improvement

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, is_best, save_dir, epoch):
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # === Load config ===
    config = load_config('configs/config.yaml')

    # === Set random seed ===
    set_seed(config['seed'])

    # === Prepare save directory ===
    os.makedirs(config['save_dir'], exist_ok=True)

    # === Logging setup ===
    log_file = config['log_file']
    log_f = open(log_file, 'a')
    log_f.write(f"\n\n========== Training started: {datetime.now()} ==========\n")

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = LaneTokenizer(nbins=config['vocab_size'] - 7)

    # === Initialize model ===
    model = Lane2Seq(
        encoder_checkpoint=config['encoder_checkpoint'],
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_size=config['ff_size'],
        max_seq_length=config['max_seq_length'],
        end_token=tokenizer.END_TOKEN
    ).to(device)

    # === Optimizer ===

    optimizer = optim.AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))

    # === Loss function ===
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # We'll refine ignore index after tokenizer

    # === Resume from checkpoint if any ===
    start_epoch = 1
    best_loss = float('inf')

    if config['resume_from_checkpoint']:
        checkpoint = torch.load(config['resume_from_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from checkpoint: {config['resume_from_checkpoint']} (epoch {start_epoch})")
        log_f.write(f"Resumed from checkpoint: {config['resume_from_checkpoint']} (epoch {start_epoch})\n")

    # === Dataset & DataLoader (will implement dataset.py next) ===
    train_dataset = TuSimpleDataset(
    root_dir=config['data_path'],
    split='train',
    nbins=config['vocab_size'] - 7,  # vocab_size includes special tokens
    format_type=config['format_type']
)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)

    assert config['vocab_size'] == train_dataset.tokenizer.vocab_size, \
    f"Config vocab_size ({config['vocab_size']}) != Tokenizer vocab_size ({train_dataset.tokenizer.vocab_size})"

    # === Training loop ===
    for epoch in range(start_epoch, config['epochs'] + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{config['epochs']}]", leave=False)

        for batch in progress_bar:
            images = batch['image'].to(device)
            #print(f"[DEBUG] Training batch image tensor shape: {images.shape}")
            target_seq = batch['target_seq'].to(device)
        
            optimizer.zero_grad()
            #print(f"Target seq: {target_seq}")
            #print(f"Target seq min: {target_seq.min().item()}, max: {target_seq.max().item()}, vocab_size: {config['vocab_size']}")
            if target_seq.max().item() >= config['vocab_size']:
                print("❌ Out of range token found in target_seq!")

            outputs = model(images, target_seq[:, :-1])  # Input sequence without last token
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_seq[:, 1:].reshape(-1))  # Predict next token
        
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())


        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")
        log_f.write(f"Epoch {epoch} - Loss: {epoch_loss:.4f}\n")

        # === Save checkpoint ===
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        if config['save_every_epoch'] or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }
            save_checkpoint(checkpoint_state, is_best, config['save_dir'], epoch)

    log_f.write(f"========== Training completed: {datetime.now()} ==========\n")
    log_f.close()
    print("Training complete!")

if __name__ == '__main__':
    main()
