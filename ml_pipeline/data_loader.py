"""
Data loader for Histopathologic Cancer Detection dataset
Handles .tif images from train_labels.csv
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
from tqdm import tqdm


class CancerDetectionDataset(Dataset):
    """
    PyTorch Dataset for cancer detection from TIF images
    """
    
    def __init__(self, 
                 image_ids: List[str],
                 labels: List[int],
                 image_dir: str,
                 transform=None):
        """
        Args:
            image_ids: List of image IDs (without .tif extension)
            labels: List of corresponding labels (0 or 1)
            image_dir: Directory containing .tif images
            transform: Optional torchvision transforms
        """
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Load image
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (96, 96), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def load_data_splits(csv_path: str,
                    image_dir: str,
                    val_split: float = 0.15,
                    test_split: float = 0.15,
                    random_seed: int = 42,
                    max_samples: int = None) -> Tuple[List, List, List, List, List, List]:
    """
    Load and split data into train/val/test sets
    
    Args:
        csv_path: Path to train_labels.csv
        image_dir: Directory containing .tif images
        val_split: Fraction for validation set
        test_split: Fraction for test set
        random_seed: Random seed for reproducibility
        max_samples: Maximum number of samples to load (for quick testing)
        
    Returns:
        Tuple of (train_ids, train_labels, val_ids, val_labels, test_ids, test_labels)
    """
    print("Loading train_labels.csv...")
    df = pd.read_csv(csv_path)
    
    # Limit samples if specified (for quick testing)
    # Limit samples if specified (for quick testing)
    if max_samples is not None and max_samples < len(df):
        # Use train_test_split to get a STRATIFIED subset
        print(f"Reducing dataset from {len(df)} to {max_samples} (Stratified)...")
        df, _ = train_test_split(
            df, 
            train_size=max_samples, 
            stratify=df['label'], 
            random_state=random_seed
        )
        print(f"Limited to {max_samples} samples for testing")
    
    image_ids = df['id'].tolist()
    labels = df['label'].tolist()
    
    print(f"Total samples: {len(image_ids)}")
    print(f"Class distribution: 0={labels.count(0)}, 1={labels.count(1)}")
    
    # First split: separate test set
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        image_ids, labels,
        test_size=test_split,
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_split / (1 - test_split)  # Adjust for already removed test set
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    print(f"Train samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    print(f"Test samples: {len(test_ids)}")
    
    return train_ids, train_labels, val_ids, val_labels, test_ids, test_labels


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Get image transformations for training or inference
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    if augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(csv_path: str,
                      image_dir: str,
                      batch_size: int = 32,
                      val_split: float = 0.15,
                      test_split: float = 0.15,
                      num_workers: int = 4,
                      max_samples: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        csv_path: Path to train_labels.csv
        image_dir: Directory containing .tif images
        batch_size: Batch size for dataloaders
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of worker processes for data loading
        max_samples: Maximum samples to use (for testing)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and split data
    train_ids, train_labels, val_ids, val_labels, test_ids, test_labels = load_data_splits(
        csv_path, image_dir, val_split, test_split, max_samples=max_samples
    )
    
    # Create datasets
    train_dataset = CancerDetectionDataset(
        train_ids, train_labels, image_dir, 
        transform=get_transforms(augment=True)
    )
    
    val_dataset = CancerDetectionDataset(
        val_ids, val_labels, image_dir,
        transform=get_transforms(augment=False)
    )
    
    test_dataset = CancerDetectionDataset(
        test_ids, test_labels, image_dir,
        transform=get_transforms(augment=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    csv_path = "../data/train_labels.csv"
    image_dir = "../data/train"
    
    print("Testing data loader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path, image_dir, batch_size=8, max_samples=1000
    )
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print("\nData loader test successful!")
