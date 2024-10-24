import torch
import torch.nn as nn
from transformer_layers import LinformerAttention

class SignLanguageEncoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        """
        Encoder class for transforming input sequences (text embeddings) into a learned representation.
        
        d_model: Dimension of model embedding.
        vocab_size: Size of the input vocabulary.
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads for multi-head attention.
        dim_feedforward: Size of feedforward network in the encoder.
        dropout: Dropout rate for regularization.
        """
        super(SignLanguageEncoder, self).__init__()

        # Embedding layer for input text sequences
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder layers
        self.layers = nn.ModuleList([
            SignLanguageEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Forward pass of the encoder.
        
        src: Input sequence (text embeddings).
        src_mask: Mask for the input sequence (optional).
        """
        # Embed the input sequence (text embeddings)
        src = self.dropout(self.embedding(src))
        
        # Pass through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class SignLanguageEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        """
        Single layer of the encoder incorporating Linformer attention.
        
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Size of the feedforward network.
        dropout: Dropout rate for regularization.
        """
        super(SignLanguageEncoderLayer, self).__init__()

        # Linformer attention mechanism for self-attention in the encoder
        self.self_attn = LinformerAttention(d_model, num_heads)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Forward pass of a single encoder layer.
        
        src: Input sequence (text embeddings).
        src_mask: Mask for the input sequence.
        """
        # Self-attention (using Linformer)
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))

        # Feedforward network
        src2 = self.feedforward(src)
        src = self.norm2(src + self.dropout(src2))

        return src
