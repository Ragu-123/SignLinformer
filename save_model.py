# save_model.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, model_name="signgennet_model"):
    """
    Save a model checkpoint, including the model state, optimizer state, and current epoch/loss.
    
    :param model: the PyTorch model to save
    :param optimizer: optimizer used during training
    :param epoch: current epoch number
    :param loss: current loss value
    :param checkpoint_dir: directory where checkpoints should be saved
    :param model_name: name of the saved model file (optional)
    """
    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Define the checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pt")
    
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, load_optimizer=True):
    """
    Load a model checkpoint and restore model state and optionally optimizer state.
    
    :param checkpoint_path: path to the saved model checkpoint
    :param model: the PyTorch model to load the state into
    :param optimizer: the optimizer to restore (optional)
    :param load_optimizer: whether to load the optimizer state as well
    :return: model, optimizer, epoch, loss
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally load optimizer state
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the loaded epoch and loss for resuming training
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Model checkpoint loaded from: {checkpoint_path}, epoch {epoch}, loss: {loss}")
    
    return model, optimizer, epoch, loss

