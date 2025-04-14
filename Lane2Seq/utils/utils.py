import os
import torch
import random
import numpy as np

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)
        print(f"✅ Saved best model to: {best_filepath}")
    else:
        print(f"💾 Checkpoint saved to: {filepath}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        print(f"✅ Loaded checkpoint (epoch {epoch})")
        return model, optimizer, epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧮 Model has {total:,} trainable parameters.")
    return total

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🪴 Random seed set to: {seed}")

def load_checkpoint_inference(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint (epoch {checkpoint.get('epoch', 0)})")
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

def denormalize_image(tensor):
    """
    Undo normalization on image tensor.
    Args:
        tensor: (C, H, W) torch.Tensor, normalized image
    Returns:
        np.array: (H, W, C), uint8 image ready for cv2
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    image = (image * std + mean)  # de-normalize
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image
