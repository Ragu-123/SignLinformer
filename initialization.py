# coding: utf-8

"""
Custom weight initialization for SignGenNet.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out

def xavier_uniform_n_(w: Tensor, gain: float = 1., n: int = 4) -> None:
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is used for RNNs or other multi-gate structures.

    :param w: parameter tensor to initialize
    :param gain: scaling factor for Xavier initialization, default is 1.
    :param n: number of matrices combined, default is 4 (for LSTMs, GRUs).
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)

def initialize_model(model: nn.Module, cfg: dict, src_padding_idx: int = None,
                     trg_padding_idx: int = None) -> None:
    """
    Initialize weights for the SignGenNet model based on the provided config.
    Handles embedding layers, Linformer attention, and transformer blocks.

    :param model: The SignGenNet model to initialize.
    :param cfg: Configuration dictionary for the model, containing initializer settings.
    :param src_padding_idx: Padding index for source embeddings (optional).
    :param trg_padding_idx: Padding index for target embeddings (optional).
    """

    # Config defaults
    gain = float(cfg.get("init_gain", 1.0))  # gain for xavier
    init = cfg.get("initializer", "xavier")  # default to xavier initialization
    init_weight = float(cfg.get("init_weight", 0.01))  # range for uniform/normal init

    embed_init = cfg.get("embed_initializer", "normal")  # default for embeddings
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))  # std for embeddings
    embed_gain = float(cfg.get("embed_init_gain", 1.0))  # gain for xavier embeddings

    bias_init = cfg.get("bias_initializer", "zeros")  # bias initializer (zeros by default)
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    # Helper function for initializers
    def _parse_init(init_type, scale, _gain):
        scale = float(scale)
        if init_type.lower() == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=_gain)
        elif init_type.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        elif init_type.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0., std=scale)
        elif init_type.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
        else:
            raise ValueError(f"Unknown initializer: {init_type}")

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    # Initialize model parameters
    with torch.no_grad():
        for name, p in model.named_parameters():
            
            # Embedding layers
            if "embed" in name:
                if "bias" in name:
                    bias_init_fn_(p)  # bias in embedding layers
                else:
                    embed_init_fn_(p)  # weight in embedding layers

            # Linformer attention or transformer layers
            elif "linformer" in name or "attention" in name:
                init_fn_(p)  # initialize Linformer/Transformer weights

            # Bias terms
            elif "bias" in name:
                bias_init_fn_(p)

            # Weights for linear layers
            elif len(p.size()) > 1:
                init_fn_(p)

        # Zero out embeddings for padding indices
        if src_padding_idx is not None and hasattr(model, 'src_embed'):
            model.src_embed.lut.weight.data[src_padding_idx].zero_()
        if trg_padding_idx is not None and hasattr(model, 'trg_embed'):
            model.trg_embed.lut.weight.data[trg_padding_idx].zero_()
