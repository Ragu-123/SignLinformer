import torch

def save_model(model, optimizer, epoch, loss, file_path):
    """
    Save the model's state, optimizer's state, and training progress.

    :param model: The trained model to save.
    :param optimizer: The optimizer being used for training.
    :param epoch: Current epoch during training.
    :param loss: The current loss at the time of saving.
    :param file_path: Path to save the model checkpoint.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, file_path)
    print(f"Model saved at {file_path}")

def load_model(file_path, model, optimizer=None):
    """
    Load the model and optimizer states from a saved checkpoint to resume training.
    
    :param file_path: Path to the saved model checkpoint.
    :param model: The model to load the state into.
    :param optimizer: The optimizer to load the state into (if resuming training).
    :return: The epoch and loss to resume training from.
    """
    checkpoint = torch.load(file_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Model loaded from {file_path}, resuming from epoch {epoch}")
    return epoch, loss
