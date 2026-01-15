"""
Utility functions for cancer detection ML pipeline
"""
import os
import numpy as np
from PIL import Image
import torch
from typing import Tuple


def load_tif_image(image_path: str) -> np.ndarray:
    """
    Load a TIF image and convert to RGB numpy array
    
    Args:
        image_path: Path to the .tif image file
        
    Returns:
        RGB numpy array of shape (H, W, 3)
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB (handles grayscale or other formats)
        img_rgb = img.convert('RGB')
        return np.array(img_rgb)
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")


def get_image_stats(image_array: np.ndarray) -> dict:
    """
    Get statistics about an image
    
    Args:
        image_array: Numpy array of image
        
    Returns:
        Dictionary with image statistics
    """
    return {
        'shape': image_array.shape,
        'dtype': image_array.dtype,
        'min': image_array.min(),
        'max': image_array.max(),
        'mean': image_array.mean(),
        'std': image_array.std()
    }


def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image_array: Numpy array of image (H, W, C)
        
    Returns:
        Normalized image array
    """
    return image_array.astype(np.float32) / 255.0


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   accuracy: float,
                   filepath: str) -> None:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   filepath: str,
                   device: str = 'cpu') -> Tuple[int, float, float]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
