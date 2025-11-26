################################################################################
# dataResNet.py
#
# Data loading and preprocessing module for ResNet50 training and evaluation.
#
# This module:
# - Defines data transformations compatible with ResNet50 (224x224, ImageNet norm)
# - Converts grayscale images to 3-channel RGB for ResNet compatibility
# - Loads training data and splits it into train/validation sets
# - Loads separate test/evaluation dataset
# - Creates PyTorch DataLoader objects for efficient batch processing
# - Applies ImageNet normalization statistics for transfer learning
#
# Key Functions:
#   get_transforms() - Returns transform pipeline for ResNet50
#   get_data_loaders() - Returns train, validation, and test DataLoaders
#
# Configuration:
#   Modify CONFIG variables at the top:
#   - DATA_DIR_TRAIN/EVAL: Paths to training and evaluation data
#   - VALIDATION_SPLIT: Fraction of training data used for validation
#   - IMAGE_SIZE: Input image size (224 for ResNet50)
################################################################################

"""
Data loading and preprocessing for ResNet50 fine-tuning
"""

from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import logging
import os

# ========================== CONFIG ==========================
DATA_DIR_TRAIN = 'Zclassification/data/in-domain-train'
DATA_DIR_EVAL = 'Zclassification/data/in-domain-eval'
IMAGE_SIZE = 224  # ResNet50 expects 224x224
BATCH_SIZE = 32
NUM_WORKERS = 0
VALIDATION_SPLIT = 0.2  # 20% of training data for validation
RANDOM_SEED = 42

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LOG_FILE = 'Zclassification/logs/data_resnet.log'
# ============================================================

# Ensure log file directory and file exist
log_dir = os.path.dirname(LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'a').close()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_transforms() -> transforms.Compose:
    """
    Get transforms for ResNet50 with ImageNet normalization.
    
    Creates a transformation pipeline that:
    - Resizes images to ResNet50's expected input size (224x224)
    - Converts grayscale images to 3-channel RGB for ResNet compatibility
    - Converts to tensor format
    - Applies ImageNet normalization statistics
    
    Returns:
        transforms.Compose: Composed transformation pipeline for image preprocessing
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def get_data_loaders(batch_size: int = BATCH_SIZE, val_split: float = VALIDATION_SPLIT) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    Load and prepare data loaders for training, validation, and testing.
    
    This function:
    - Loads the full training dataset from DATA_DIR_TRAIN
    - Splits training data into train and validation sets based on val_split
    - Loads separate test dataset from DATA_DIR_EVAL
    - Creates PyTorch DataLoader objects for each split
    - Applies the transformation pipeline to all datasets
    
    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to BATCH_SIZE.
        val_split (float, optional): Fraction of training data (0-1) to use for validation. 
                                     Defaults to VALIDATION_SPLIT (0.2).
    
    Returns:
        tuple containing:
            - train_loader (DataLoader): DataLoader for training set
            - val_loader (DataLoader): DataLoader for validation set
            - test_loader (DataLoader): DataLoader for test/evaluation set
            - num_classes (int): Number of classes in the dataset
            - class_names (list[str]): List of class names sorted alphabetically
    
    Raises:
        FileNotFoundError: If data directories don't exist
        ValueError: If val_split is not between 0 and 1
    """
    logger.info("Loading datasets...")
    
    transform = get_transforms()
    
    # Load full training dataset
    full_train_data = datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform)
    num_classes = len(full_train_data.classes)
    class_names = full_train_data.classes
    
    logger.info(f"Total training samples: {len(full_train_data)}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Split training data into train and validation
    torch.manual_seed(RANDOM_SEED)
    val_size = int(len(full_train_data) * val_split)
    train_size = len(full_train_data) - val_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])
    
    logger.info(f"Training samples: {train_size}")
    logger.info(f"Validation samples: {val_size}")
    
    # Load test dataset
    test_data = datasets.ImageFolder(DATA_DIR_EVAL, transform=transform)
    logger.info(f"Test samples: {len(test_data)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, num_classes, class_names


if __name__ == '__main__':
    # Test data loading
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders()
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    logger.info(f"Sample batch shape: {images.shape}")
    logger.info(f"Sample labels shape: {labels.shape}")
