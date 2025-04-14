# lane2seq_project/models/lane2seq.py

import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder import TransformerDecoder

class Lane2Seq(nn.Module):
    def __init__(self, encoder_checkpoint: str, vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8, ff_size=1024, max_seq_length=512, end_token=2):
        super(Lane2Seq, self).__init__()

        # Encoder: ViT-MAE initialized
        self.encoder = ViTEncoder(checkpoint_path=encoder_checkpoint)

        self.encoder_to_decoder = nn.Linear(768, hidden_size)  # 768 from ViT encoder, hidden_size from decoder config

        # Decoder: Transformer
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            max_seq_length=max_seq_length
        )
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
    
        # ✅ Pass tokenizer value directly
        self.END_TOKEN = end_token

        print(f"[Info] END_TOKEN id is: {self.END_TOKEN}")

    def forward(self, images, target_seq):
        """
        Args:
            images: Tensor (batch_size, 3, H, W)
            target_seq: Tensor (batch_size, seq_length) - tokenized target sequence

        Returns:
            logits: Tensor (batch_size, seq_length, vocab_size)
        """
        # Encode images
        encoder_outputs = self.encoder(images)  # (batch_size, seq_len, hidden_size)
        encoder_outputs = self.encoder_to_decoder(encoder_outputs)  # [batch, seq_len, hidden_size]


        # Prepare causal mask for decoder
        seq_length = target_seq.size(1)
        tgt_mask = self.decoder.get_tgt_mask(seq_length)

        # Decode target sequence
        logits = self.decoder(tgt_seq=target_seq, memory=encoder_outputs, tgt_mask=tgt_mask)

        return logits

    def generate(self, images, prompt_token, max_length=None):
        """
        Inference autoregressive token generation.

        Args:
            images: Tensor (batch_size, 3, H, W)
            prompt_token: Tensor (batch_size, prompt_seq_len)
            max_length: int, optional (default: self.max_seq_length)

        Returns:
            generated: Tensor (batch_size, generated_seq_len)
        """
        self.eval()
        device = images.device
        batch_size = images.size(0)

        # Encode image
        encoder_outputs = self.encoder(images)
        encoder_outputs = self.encoder_to_decoder(encoder_outputs)

        # Initialize generated sequence with prompt
        generated = prompt_token.clone()  # (B, seq_len)

        if max_length is None:
            max_length = self.max_seq_length

        for _ in range(max_length - prompt_token.size(1)):
            seq_length = generated.size(1)

            # Prepare mask
            tgt_mask = self.decoder.get_tgt_mask(seq_length).to(device)

            # Forward through decoder
            logits = self.decoder(
                tgt_seq=generated,
                memory=encoder_outputs,
                tgt_mask=tgt_mask
            )  # (B, seq_len, vocab_size)

            # Take last token's logits
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)

            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)

            # Append next token
            generated = torch.cat([generated, next_tokens], dim=1)  # (B, seq_len + 1)

            # Early stopping: check END_TOKEN
            if (next_tokens == self.END_TOKEN).all():
                break

        return generated
