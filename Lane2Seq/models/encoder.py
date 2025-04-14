# lane2seq_project/models/encoder.py

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
from safetensors.torch import load_file as safe_load
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super(ViTEncoder, self).__init__()
    
        # Load ViT-Base config
        self.config = ViTConfig.from_pretrained(checkpoint_path)
    
        # ✅ Explicitly set image sizes
        self.config.image_size = (320, 800)  # (height, width)
        self.config.patch_size = 16
        self.config.hidden_size = 768  # ViT-Base hidden size
    
        # Initialize ViT model
        self.vit = ViTModel(self.config)
    
        # Load safetensor weights
        state_dict = safe_load(f"{checkpoint_path}/model.safetensors")
    
        # Clean state dict keys
        state_dict = self.clean_state_dict(state_dict)
    
        # Interpolate position embeddings if needed
        state_dict = self.interpolate_positional_embeddings(state_dict)
    
        # Debugging info
        print(f"[Info] Model expects image size: {self.config.image_size}")
        print(f"[Info] Position embedding shape after interpolation: {state_dict['embeddings.position_embeddings'].shape}")
    
        # Load weights
        missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
    
        if missing_keys:
            print(f"[Warning] Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys in the pretrained weights: {unexpected_keys}")
    
        # Ensure encoder is trainable
        for param in self.vit.parameters():
            param.requires_grad = True


    def clean_state_dict(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vit."):
                new_key = k[len("vit."):]  # remove 'vit.' prefix
                new_state_dict[new_key] = v
        return new_state_dict

    def interpolate_positional_embeddings(self, state_dict):
        if "embeddings.position_embeddings" not in state_dict:
            return state_dict  # No position embeddings found
    
        pos_embed = state_dict["embeddings.position_embeddings"]  # (1, old_seq_len, hidden_dim)
    
        # ✅ Use model config image size dynamically
        new_height, new_width = self.config.image_size
        patch_size = self.config.patch_size
    
        num_patches = (new_height // patch_size) * (new_width // patch_size)
        new_seq_len = num_patches + 1  # +1 for cls token
    
        if pos_embed.size(1) == new_seq_len:
            return state_dict  # No interpolation needed
    
        print(f"[Info] Interpolating position embeddings from {pos_embed.size(1)} to {new_seq_len}")
    
        cls_token = pos_embed[:, :1, :]  # (1, 1, hidden_dim)
        pos_tokens = pos_embed[:, 1:, :]  # (1, old_seq_len-1, hidden_dim)
    
        # Get old grid size
        old_num_patches = pos_tokens.size(1)
        old_size = int(old_num_patches ** 0.5)
        pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)  # (1, hidden_dim, old_size, old_size)
    
        # Interpolate
        new_size = (new_height // patch_size, new_width // patch_size)
        pos_tokens = F.interpolate(pos_tokens, size=new_size, mode='bicubic', align_corners=False)
    
        # Reshape back
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, pos_tokens.size(1))
    
        # Combine with class token
        new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
    
        state_dict["embeddings.position_embeddings"] = new_pos_embed
        return state_dict


    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor of shape (batch_size, 3, H, W)
        Returns:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state
