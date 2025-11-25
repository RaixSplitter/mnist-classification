import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import get_mnist_loaders
from my_models import SimpleCNN


def train_model(num_epochs=10, batch_size=64, learning_rate=0.001, device=None, plot_loss=False):
    """
    Train the SimpleCNN model on MNIST dataset.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        learning_rate: Learning rate for Adam optimizer
        device: Device to train on (cuda/cpu). Auto-detected if None.
        plot_loss: Whether to plot training loss over epochs
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, _ = get_mnist_loaders(batch_size=batch_size, num_workers=0)
    
    # Initialize model
    model = SimpleCNN(num_classes=10)  # MNIST has 10 classes (0-9)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track loss history
    loss_history = []
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })
        
        # Print epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")
    
    # Plot training loss
    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        models_dir = Path('mnist_classification/models')
        models_dir.mkdir(parents=True, exist_ok=True)
        plot_path = models_dir / 'training_loss.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Training loss plot saved to {plot_path}")
        plt.show()
    
    # Save model
    models_dir = Path('mnist_classification/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'mnist_cnn.pth'
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    return model


if __name__ == '__main__':
    # Train the model
    train_model(num_epochs=5, batch_size=64, learning_rate=0.001, plot_loss=True)
