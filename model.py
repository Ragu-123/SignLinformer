import torch
import torch.nn as nn
from encoder import SignLanguageEncoder
from decoder import SignLanguageDecoder

class SignLanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_encoder_layers, num_decoder_layers, 
                 num_heads, dim_feedforward, keypoint_dim, dropout=0.1):
        """
        Full Transformer-based model for sign language generation using Linformer attention.

        vocab_size: Size of the input vocabulary.
        d_model: Model embedding dimension.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads for multi-head attention.
        dim_feedforward: Size of the feedforward network.
        keypoint_dim: Dimensionality of the output keypoints.
        dropout: Dropout rate for regularization.
        """
        super(SignLanguageTransformer, self).__init__()

        # Embedding for text input
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Encoder to process text input
        self.encoder = SignLanguageEncoder(d_model, vocab_size, num_encoder_layers, num_heads, dim_feedforward, dropout)

        # Decoder to generate pose keypoints
        self.decoder = SignLanguageDecoder(d_model, keypoint_dim, num_decoder_layers, num_heads, dim_feedforward, dropout)

        # Final linear layer to map decoder output to pose keypoints
        self.output_linear = nn.Linear(d_model, keypoint_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_text, tgt_keypoints, src_mask=None, tgt_mask=None):
        """
        Forward pass of the model.
        
        src_text: Input text sequence.
        tgt_keypoints: Target keypoints sequence.
        src_mask: Mask for the input text sequence.
        tgt_mask: Mask for the target keypoints sequence.
        """
        # Embed the input text
        src_emb = self.dropout(self.embedding(src_text))

        # Pass through the encoder
        memory = self.encoder(src_emb, src_mask)

        # Pass through the decoder
        output = self.decoder(tgt_keypoints, memory, tgt_mask, src_mask)

        # Map decoder output to pose keypoints
        keypoint_output = self.output_linear(output)

        return keypoint_output

