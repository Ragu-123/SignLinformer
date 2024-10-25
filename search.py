# coding: utf-8

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from decoders import Decoder, TransformerDecoder
from embeddings import Embeddings

def greedy(
        src_mask: Tensor,
        embed: Embeddings,
        decoder: Decoder,
        encoder_output: Tensor,
        trg_input: Tensor,
        model,
        ) -> (np.array, np.array):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: Mask for source inputs, 0 for positions after </s>
    :param embed: Target embedding
    :param decoder: Decoder to use for greedy decoding
    :param encoder_output: Encoder hidden states for attention
    :param trg_input: Target input (ground truth)
    :param model: Model instance (contains specific configurations)
    :return:
        - stacked_output: Output hypotheses (2D array of indices),
        - stacked_attention_scores: Attention scores (3D array, can be None)
    """
    # Initialize the input: extract just the BOS first frame from the target
    ys = trg_input[:, :1, :].float()
    ys_out = ys  # This will hold the output sequences

    # Set the target mask by finding the padded rows
    trg_mask = trg_input != 0.0
    trg_mask = trg_mask.unsqueeze(1)  # Adding a dimension for broadcasting

    max_output_length = trg_input.shape[1]  # Find the maximum output length for this batch

    if model.just_count_in:
        ys = ys[:, :, -1:]  # If just counting in, keep the last frame

    for i in range(max_output_length):
        # Update ys for the current iteration
        if model.just_count_in:
            ys[:, -1] = trg_input[:, i, -1:]  # Drive the input using the GT counter
        else:
            ys[:, -1, -1:] = trg_input[:, i, -1:]  # Timing driven by GT counter

        # Embed the target input before passing to the decoder
        trg_embed = embed(ys)

        # Create padding mask to the required size
        padding_mask = trg_mask[:, :, :i + 1, :i + 1]
        pad_amount = padding_mask.shape[2] - padding_mask.shape[3]
        padding_mask = (F.pad(input=padding_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)

        # Pass the embedded input and the encoder output into the decoder
        with torch.no_grad():
            out, _, _, _ = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                src_mask=src_mask,
                trg_mask=padding_mask,
            )

            if model.future_prediction != 0:
                # Cut to only the first frame prediction
                out = torch.cat((out[:, :, :out.shape[2] // model.future_prediction], out[:, :, -1:]), dim=2)

            if model.just_count_in:
                # If just counter in trg_input, concatenate counters of output
                ys = torch.cat([ys, out[:, -1:, -1:]], dim=1)

            # Add this frame prediction to the overall prediction
            ys = torch.cat([ys, out[:, -1:, :]], dim=1)
            # Add this next predicted frame to the full output
            ys_out = torch.cat([ys_out, out[:, -1:, :]], dim=1)

    return ys_out, None  # Returning None for stacked_attention_scores as not utilized
