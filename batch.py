# coding: utf-8

"""
Implementation of a mini-batch.
"""

import torch
import torch.nn.functional as F

from constants import TARGET_PAD


class Batch:
    """Object for holding a batch of data with masks during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, model):
        """
        Create a new batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        lengths, masks, and the number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch: The input batch from the torch text iterator.
        :param pad_index: Index for the padding token.
        :param model: The model instance used for training.
        """
        # Unpack source sequences and their lengths
        self.src, self.src_lengths = torch_batch.src
        # Create a mask where the source is not padding
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)  # Number of sequences in the batch
        self.trg_input = None  # Placeholder for target input
        self.trg = None  # Placeholder for target sequences
        self.trg_mask = None  # Placeholder for target mask
        self.trg_lengths = None  # Lengths of target sequences
        self.ntokens = None  # Number of non-padding tokens in target

        # File paths for tracking or saving data
        self.file_paths = torch_batch.file_paths
        self.use_cuda = model.use_cuda  # Check if using GPU
        self.target_pad = TARGET_PAD  # Padding token for the target sequences

        # Just Count option for model behavior
        self.just_count_in = model.just_count_in
        # Future Prediction option for model behavior
        self.future_prediction = model.future_prediction

        # If the target (trg) is provided in the torch batch
        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = trg.shape[1]  # Length of target sequences

            # trg_input is used for teacher forcing, last one is cut off
            # Remove the last frame for target input, as inputs are only up to frame N-1
            self.trg_input = trg.clone()[:, :-1, :]

            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg.clone()[:, 1:, :]

            # Just Count: If set, cut off the first frame of trg_input
            if self.just_count_in:
                self.trg_input = self.trg_input[:, :, -1:]

            # Future Prediction: Create future target sequences
            if self.future_prediction != 0:
                future_trg = torch.Tensor()  # Initialize tensor for future targets
                # Concatenate future frames to create the target
                for i in range(0, self.future_prediction):
                    future_trg = torch.cat((future_trg, self.trg[:, i:-(self.future_prediction - i), :-1].clone()), dim=2)
                # Create the final target using the collected future_trg and original trg
                self.trg = torch.cat((future_trg, self.trg[:, :-self.future_prediction, -1:]), dim=2)

                # Cut off the last N frames of the trg_input for future predictions
                self.trg_input = self.trg_input[:, :-self.future_prediction, :]

            # Create a dynamic target mask excluding the padded areas from the loss computation
            trg_mask = (self.trg_input != self.target_pad).unsqueeze(1)
            # Pad the target mask to ensure proper dimensions
            pad_amount = self.trg_input.shape[1] - self.trg_input.shape[2]
            self.trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
            # Count non-padding tokens in trg for loss calculation
            self.ntokens = (self.trg != pad_index).data.sum().item()

        # Move the batch to GPU if applicable
        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU.

        :return: None
        """
        self.src = self.src.cuda()  # Move source to GPU
        self.src_mask = self.src_mask.cuda()  # Move source mask to GPU

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()  # Move target input to GPU
            self.trg = self.trg.cuda()  # Move target to GPU
            self.trg_mask = self.trg_mask.cuda()  # Move target mask to GPU
