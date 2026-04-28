"""Data loading and preprocessing module."""
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data loading, preprocessing, and augmentation."""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.DATASET_NAME
        self.batch_size = config.BATCH_SIZE
        self.image_size = config.IMAGE_SIZE
        
        # Define transforms with augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # No augmentation for validation/test
        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    
    def load_cifar100(self):
        """Load CIFAR-100 dataset."""
        logger.info("Loading CIFAR-100 dataset...")
        
        # Download and load training data
        train_dataset = CIFAR100(
            root=str(self.config.DATA_DIR),
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Download and load test data
        test_dataset = CIFAR100(
            root=str(self.config.DATA_DIR),
            train=False,
            download=True,
            transform=self.val_transform
        )
        
        # Split training data into train and validation
        train_size = int(len(train_dataset) * (1 - self.config.VAL_SPLIT))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size]
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self):
        """Get train, validation, and test dataloaders."""
        train_dataset, val_dataset, test_dataset = self.load_cifar100()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def analyze_dataset(dataloader, name="Dataset"):
        """Analyze dataset statistics."""
        logger.info(f"\n=== {name} Analysis ===")
        logger.info(f"Batch size: {dataloader.batch_size}")
        logger.info(f"Total batches: {len(dataloader)}")
        logger.info(f"Total samples: {len(dataloader.dataset)}")
        
        # Calculate mean and std
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_images = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= 10:  # Calculate from first 10 batches for speed
                break
            b, c, h, w = images.shape
            images = images.view(b, c, -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += b
        
        mean /= total_images
        std /= total_images
        
        logger.info(f"Mean: {mean}")
        logger.info(f"Std: {std}")
