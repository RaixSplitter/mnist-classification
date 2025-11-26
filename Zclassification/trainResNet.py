################################################################################
# trainResNet.py
#
# Training script for fine-tuning ResNet50 on custom image classification task.
#
# This script:
# - Loads pretrained ResNet50 with ImageNet weights
# - Fine-tunes the model on training data with validation monitoring
# - Supports freezing backbone layers (transfer learning) or full fine-tuning
# - Tracks and logs training/validation loss and accuracy per epoch
# - Saves the best model based on validation accuracy
# - Generates training history plots (loss and accuracy curves)
# - Implements early stopping based on validation performance
#
# Usage:
#   python trainResNet.py
#
# Configuration:
#   Modify CONFIG variables at the top of the script:
#   - NUM_EPOCHS: Number of training epochs
#   - BATCH_SIZE: Training batch size
#   - LEARNING_RATE: Optimizer learning rate
#   - FREEZE_BACKBONE: Whether to freeze pretrained layers
################################################################################

"""
Training script for ResNet50 fine-tuning
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import logging
import os

from dataResNet import get_data_loaders

# ========================== CONFIG ==========================
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FREEZE_BACKBONE = True  # If True, only train final layer

MODEL_SAVE_PATH = 'Zclassification/models/resnet50_finetuned.pth'
PLOT_SAVE_PATH = 'Zclassification/plots/training_loss_resnet50.png'
LOG_FILE = 'Zclassification/logs/train_resnet.log'
# ============================================================

# Create directories
os.makedirs('Zclassification/models', exist_ok=True)
os.makedirs('Zclassification/plots', exist_ok=True)
os.makedirs('Zclassification/logs', exist_ok=True)

# Ensure log file directory exists
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


def create_model(num_classes: int, freeze_backbone: bool = FREEZE_BACKBONE) -> nn.Module:
    """
    Create ResNet50 model with pretrained ImageNet weights.
    
    This function:
    - Loads ResNet50 architecture with IMAGENET1K_V2 pretrained weights
    - Optionally freezes all backbone layers for transfer learning
    - Replaces the final fully connected layer to match num_classes
    - Prepares model for fine-tuning on custom classification task
    
    Args:
        num_classes (int): Number of output classes for the classification task
        freeze_backbone (bool, optional): If True, freezes all pretrained layers except 
                                         the final FC layer. If False, all layers are 
                                         trainable. Defaults to FREEZE_BACKBONE.
    
    Returns:
        nn.Module: ResNet50 model ready for training/fine-tuning
        
    Note:
        When freeze_backbone=True, only the final FC layer parameters will be trainable,
        which is useful for transfer learning with limited data.
    """
    logger.info("Loading pretrained ResNet50...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    
    # Freeze backbone if specified
    if freeze_backbone:
        logger.info("Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    logger.info(f"Replaced final layer: {num_features} -> {num_classes}")
    
    return model


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device, 
                epoch: int, num_epochs: int) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Performs one complete pass through the training dataset:
    - Sets model to training mode
    - Iterates through all training batches
    - Computes forward pass, loss, and backward pass
    - Updates model parameters via optimizer
    - Tracks running loss and accuracy
    - Displays progress bar with real-time metrics
    
    Args:
        model (nn.Module): Neural network model to train
        train_loader (DataLoader): DataLoader containing training data
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates (e.g., Adam)
        device (torch.device): Device to run training on (cuda or cpu)
        epoch (int): Current epoch number (0-indexed)
        num_epochs (int): Total number of epochs for training
    
    Returns:
        tuple containing:
            - epoch_loss (float): Average loss over all batches in the epoch
            - epoch_acc (float): Training accuracy (%) for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / (i + 1)
        accuracy = 100 * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device, epoch: int, num_epochs: int) -> Tuple[float, float]:
    """
    Validate model for one epoch.
    
    Performs evaluation on the validation dataset:
    - Sets model to evaluation mode (disables dropout, batchnorm updates)
    - Iterates through all validation batches without gradient computation
    - Computes loss and accuracy metrics
    - Displays progress bar with real-time metrics
    
    Args:
        model (nn.Module): Neural network model to validate
        val_loader (DataLoader): DataLoader containing validation data
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        device (torch.device): Device to run validation on (cuda or cpu)
        epoch (int): Current epoch number (0-indexed)
        num_epochs (int): Total number of epochs for training
    
    Returns:
        tuple containing:
            - val_loss (float): Average loss over all validation batches
            - val_acc (float): Validation accuracy (%) for the epoch
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
            accuracy = 100 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    val_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def plot_training_history(train_losses: List[float], val_losses: List[float], 
                          train_accs: List[float], val_accs: List[float]) -> None:
    """
    Plot and save training history graphs.
    
    Creates a two-panel figure showing:
    - Left panel: Training and validation loss over epochs
    - Right panel: Training and validation accuracy over epochs
    
    The plot is saved to PLOT_SAVE_PATH and displayed to the user.
    
    Args:
        train_losses (list[float]): List of training losses for each epoch
        val_losses (list[float]): List of validation losses for each epoch
        train_accs (list[float]): List of training accuracies (%) for each epoch
        val_accs (list[float]): List of validation accuracies (%) for each epoch
    
    Returns:
        None
        
    Side Effects:
        - Saves plot to PLOT_SAVE_PATH as PNG file
        - Displays plot in GUI window
        - Logs save location to logger
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-o', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    logger.info(f"Training plot saved to {PLOT_SAVE_PATH}")
    plt.show()


def train_model(num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, 
                learning_rate: float = LEARNING_RATE) -> Tuple[nn.Module, List[float], List[float], List[float], List[float]]:
    """
    Main training function for ResNet50 fine-tuning.
    
    Complete training pipeline that:
    - Loads and prepares training/validation/test data
    - Creates and configures ResNet50 model
    - Trains model for specified number of epochs
    - Validates after each epoch
    - Saves best model based on validation accuracy
    - Plots and saves training history
    - Logs all metrics and progress
    
    Args:
        num_epochs (int, optional): Number of training epochs. Defaults to NUM_EPOCHS.
        batch_size (int, optional): Batch size for training and validation. Defaults to BATCH_SIZE.
        learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to LEARNING_RATE.
    
    Returns:
        tuple containing:
            - model (nn.Module): Trained model (from last epoch, not necessarily best)
            - train_losses (list[float]): Training loss for each epoch
            - val_losses (list[float]): Validation loss for each epoch
            - train_accs (list[float]): Training accuracy (%) for each epoch
            - val_accs (list[float]): Validation accuracy (%) for each epoch
    
    Side Effects:
        - Saves best model checkpoint to MODEL_SAVE_PATH
        - Saves training history plot to PLOT_SAVE_PATH
        - Logs extensive training information to LOG_FILE
    
    Note:
        The best model (highest validation accuracy) is saved to disk.
        Use evaluationResNet.py to load and evaluate the best model.
    """
    logger.info("="*60)
    logger.info("Starting ResNet50 Fine-tuning")
    logger.info("="*60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Freeze backbone: {FREEZE_BACKBONE}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(batch_size)
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Setup optimizer and criterion
    if FREEZE_BACKBONE:
        optimizer = Adam(model.fc.parameters(), lr=learning_rate)
        logger.info("Optimizing only final layer parameters")
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate)
        logger.info("Optimizing all model parameters")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch, num_epochs)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': num_classes,
                'class_names': class_names
            }, MODEL_SAVE_PATH)
            logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {MODEL_SAVE_PATH}")
    logger.info("="*60)
    
    return model, train_losses, val_losses, train_accs, val_accs


if __name__ == '__main__':
    train_model() 