import torch
import torch.nn as nn
from linformer import Linformer

class SignLanguageEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, linformer_config):
        """
        Encoder to process text into a meaningful latent representation using Linformer-based attention.

        vocab_size: Size of the vocabulary.
        embedding_dim: Dimensionality of the word embeddings.
        linformer_config: Configuration for Linformer layers.
        """
        super(SignLanguageEncoder, self).__init__()
        
        # Embedding layer for converting tokens into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linformer layer for attention
        self.linformer = Linformer(
            dim=embedding_dim,
            seq_len=linformer_config["seq_len"],
            depth=linformer_config["depth"],
            k=linformer_config["k"],
            heads=linformer_config["heads"],
            dropout=linformer_config["dropout"]
        )
        
        # Optional linear layer to project the output into another dimension
        self.fc = nn.Linear(embedding_dim, linformer_config["hidden_dim"])

    def forward(self, input_ids):
        """
        Forward pass of the encoder.
        
        input_ids: Tensor of tokenized input sentence (batch_size, seq_len)
        """
        # Convert token ids to embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Apply Linformer attention
        encoded_output = self.linformer(embeddings)  # (batch_size, seq_len, hidden_dim)
        
        # Optional linear projection
        output = self.fc(encoded_output)  # (batch_size, seq_len, hidden_dim)

        return output
