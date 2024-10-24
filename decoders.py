import torch
import torch.nn as nn
from linformer import Linformer

class SignLanguageDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, linformer_config):
        """
        Decoder for Sign Language generation using Linformer-based transformer architecture.
        
        output_dim: Dimensionality of the final output (number of keypoints per frame * 3 for x, y, and confidence values).
        embedding_dim: Dimension of the input embeddings.
        linformer_config: Configuration dictionary for Linformer, including sequence length, number of heads, and hidden dimensions.
        """
        super(SignLanguageDecoder, self).__init__()
        
        # Linformer-based decoder layers
        self.linformer = Linformer(
            dim=embedding_dim,
            seq_len=linformer_config['seq_len'],
            depth=linformer_config['depth'],
            k=linformer_config['k'],
            heads=linformer_config['heads'],
            dropout=linformer_config['dropout']
        )
        
        # Final linear layer to map Linformer output to pose keypoints
        self.fc_out = nn.Linear(embedding_dim, output_dim)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(linformer_config['dropout'])
        
    def forward(self, encoder_output):
        """
        Forward pass of the decoder.
        
        encoder_output: Output from the encoder (batch_size, seq_len, embedding_dim).
        """
        # Pass the encoder output through the Linformer-based transformer layers
        linformer_out = self.linformer(encoder_output)
        
        # Apply dropout
        linformer_out = self.dropout(linformer_out)
        
        # Generate the final output representing the predicted pose keypoints
        output = self.fc_out(linformer_out)
        
        return output
