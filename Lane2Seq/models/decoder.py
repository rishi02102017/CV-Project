# lane2seq_project/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8, ff_size=1024, max_seq_length=512):
        super(TransformerDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # Learnable start and end tokens
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.end_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output linear layer to vocab
        self.output_head = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, tgt_seq, memory, tgt_mask=None):
        """
        Args:
            tgt_seq: Tensor (batch_size, seq_length) - token ids of target sequence
            memory: Tensor (batch_size, src_seq_len, hidden_size) - encoder outputs
            tgt_mask: Tensor (seq_length, seq_length) - causal mask for decoder

        Returns:
            logits: Tensor (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = tgt_seq.size()

        # Token and position embeddings
        token_emb = self.token_embedding(tgt_seq)  # (batch_size, seq_length, hidden_size)

        positions = torch.arange(seq_length, device=tgt_seq.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_emb = self.position_embedding(positions)

        # Add token and position embeddings
        tgt_emb = token_emb + pos_emb

        # Pass through Transformer Decoder
        output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

        # Project to vocab size
        logits = self.output_head(output)

        return logits

    def get_tgt_mask(self, seq_length):
        """
        Generate causal mask for decoder (prevent attending to future tokens).
        """
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)
