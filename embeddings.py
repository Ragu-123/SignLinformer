import torch
import math
from torch import nn, Tensor

class Embeddings(nn.Module):
    """
    Embedding class for handling token embeddings and optional positional encodings, 
    designed for transformer models with scaling and freezing capabilities.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, padding_idx: int = 1,
                 max_len: int = 5000, scale: bool = False, freeze: bool = False):
        """
        Initializes the embedding layer with optional positional encoding and scaling.

        :param vocab_size: Size of the vocabulary (number of unique tokens).
        :param embedding_dim: Dimensionality of the embedding vectors.
        :param padding_idx: Index for the padding token in the vocabulary.
        :param max_len: Maximum length of the input sequence for positional encodings.
        :param scale: If True, scale embeddings by sqrt(embedding_dim) (used in Transformers).
        :param freeze: If True, embeddings will not be updated during training.
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # Positional encoding
        self.position_embedding = nn.Parameter(self._create_positional_encoding(max_len, embedding_dim), requires_grad=False)

        # Freeze embedding weights if required
        if freeze:
            self.lut.weight.requires_grad = False

    def _create_positional_encoding(self, max_len: int, embedding_dim: int) -> Tensor:
        """
        Creates positional encodings for input sequences, following the sine and cosine formula.

        :param max_len: Maximum length of the input sequence.
        :param embedding_dim: Dimensionality of the embedding vectors.
        :return: A tensor of positional encodings with shape (max_len, embedding_dim).
        """
        pos_enc = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc.unsqueeze(0)  # Add batch dimension

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the embedding layer.

        :param x: Input tensor of token indices (batch_size, sequence_length)
        :return: Embedded tensor (batch_size, sequence_length, embedding_dim)
        """
        token_embeddings = self.lut(x)  # Lookup token embeddings
        
        # Optionally scale embeddings by sqrt(embedding_dim)
        if self.scale:
            token_embeddings = token_embeddings * math.sqrt(self.embedding_dim)

        # Add positional encodings
        seq_len = x.size(1)
        position_embeddings = self.position_embedding[:, :seq_len, :]

        return token_embeddings + position_embeddings

    def __repr__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size})"
