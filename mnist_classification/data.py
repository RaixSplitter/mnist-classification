import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def download_mnist(data_dir='mnist_classification/data'):
    """Download MNIST dataset and organize it for ImageFolder"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download training data
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True
    )
    
    # Download test data
    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True
    )
    
    return data_path


def get_transforms():
    """Define transforms: resize, grayscale, to tensor, z-standardize"""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # MNIST is already 28x28, but including for consistency
        transforms.Grayscale(num_output_channels=1),  # MNIST is already grayscale
        transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Z-standardize: (x - 0.5) / 0.5 -> [-1, 1]
    ])
    return transform


def get_mnist_loaders(data_dir='mnist_classification/data', batch_size=32, num_workers=0):
    """
    Get MNIST data loaders using ImageFolder and transforms.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, test_loader
    """
    # Download MNIST data first
    data_path = download_mnist(data_dir)
    
    # Get transforms
    transform = get_transforms()
    
    # Load datasets with transforms
    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        transform=transform,
        download=False  # Already downloaded
    )
    
    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        transform=transform,
        download=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the data loading
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
