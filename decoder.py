import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_layers import LinformerAttention

class SignLanguageDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, dim_feedforward, dropout=0.1):
        """
        Decoder class for transforming the encoded representations back into sign language pose keypoints.
        
        d_model: Dimension of model embedding.
        vocab_size: Size of the output vocabulary.
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads for multi-head attention.
        dim_feedforward: Size of feedforward network in the decoder.
        dropout: Dropout rate for regularization.
        """
        super(SignLanguageDecoder, self).__init__()

        # Embedding for output sequences (pose keypoints predicted back to embeddings)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            SignLanguageDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Linear layer for predicting pose keypoints
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_output, trg_mask=None, memory_mask=None):
        """
        Forward pass of the decoder.
        
        trg: Target sequence input (pose keypoints).
        enc_output: Output from the encoder (context vectors).
        trg_mask: Mask for the target sequence (optional).
        memory_mask: Mask for the encoder output (optional).
        """
        # Embed target (pose keypoints) sequence
        trg = self.dropout(self.embedding(trg))
        
        # Pass through the decoder layers
        for layer in self.layers:
            trg = layer(trg, enc_output, trg_mask, memory_mask)

        # Predict the output keypoints
        output = self.fc_out(trg)
        return output

class SignLanguageDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        """
        Single layer of the decoder which incorporates Linformer attention.
        
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Size of the feedforward network.
        dropout: Dropout rate for regularization.
        """
        super(SignLanguageDecoderLayer, self).__init__()

        # Linformer attention mechanism for decoder self-attention and encoder-decoder attention
        self.self_attn = LinformerAttention(d_model, num_heads)
        self.multihead_attn = LinformerAttention(d_model, num_heads)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_output, trg_mask=None, memory_mask=None):
        """
        Forward pass of a single decoder layer.
        
        trg: Target sequence input (pose keypoints).
        enc_output: Output from the encoder.
        trg_mask: Mask for the target sequence.
        memory_mask: Mask for the encoder output.
        """
        # Self-attention (using Linformer)
        trg2 = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(trg2))

        # Encoder-decoder attention (Linformer)
        trg2 = self.multihead_attn(trg, enc_output, enc_output, memory_mask)
        trg = self.norm2(trg + self.dropout(trg2))

        # Feedforward network
        trg2 = self.feedforward(trg)
        trg = self.norm3(trg + self.dropout(trg2))

        return trg
