# coding: utf-8
"""
Module to represent whole models
"""

import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from initialization import initialize_model
from embeddings import Embeddings
from encoders import Encoder, TransformerEncoder
from decoders import Decoder, TransformerDecoder
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from search import greedy
from vocabulary import Vocabulary
from batch import Batch

class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int,
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.trg_embed = trg_embed

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in", True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in", False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise", False)
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)

    # pylint: disable=arguments-differ
    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                src_mask: Tensor,
                src_lengths: Tensor,
                trg_mask: Tensor = None,
                src_input: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)

        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:, :, -1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds, torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate * noise

        # Decode the target sequence
        skel_out, dec_hidden, _, _ = self.decode(encoder_output=encoder_output,
                                                 src_mask=src_mask, trg_input=trg_input,
                                                 trg_mask=trg_mask)

        gloss_out = None

        return skel_out, gloss_out

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)

        return encode_output

    def decode(self, encoder_output: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               trg_mask: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        # Embed the target using a linear layer
        trg_embed = self.trg_embed(trg_input)
        # Apply decoder to the embedded target
        decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                                      src_mask=src_mask, trg_mask=trg_mask)

        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg.mask)

        # Compute batch loss using skel_out and the batch target
        batch_loss = loss_function(skel_out, batch.trg)

        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None

        # Return batch loss = sum over all elements in batch that are not pad
        return batch_loss, noise

    def run_batch(self, batch: Batch, max_output_length: int) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src.mask)

        # If maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # Then decode the batch separately, as needs to be done iteratively
        # Greedy decoding
        stacked_output, stacked_attention_scores = greedy(
            encoder_output=encoder_output,
            src_mask=batch.src.mask,
            embed=self.trg_embed,
            decoder=self.decoder,
            trg_input=batch.trg_input,
            model=self)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = trg_vocab.size() + trg_vocab.size() + 1
    out_trg_size = trg_vocab.size()

    encoder = TransformerEncoder(cfg)
    decoder = TransformerDecoder(cfg)

    # Input embedding
    src_embed = Embeddings(vocab=src_vocab, d_model=cfg["d_model"], padding_idx=src_padding_idx)

    # Target embedding
    trg_embed = Embeddings(vocab=trg_vocab, d_model=cfg["d_model"], padding_idx=trg_padding_idx)

    model = Model(encoder, decoder, src_embed, trg_embed,
                  src_vocab, trg_vocab, cfg, in_trg_size, out_trg_size)

    # Initialize the model
    initialize_model(model, cfg)

    return model
