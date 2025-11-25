import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_mnist_loaders
from my_models import SimpleCNN


def evaluate_model(model_path='mnist_classification/models/mnist_cnn.pth', 
                   batch_size=64, 
                   device=None,
                   show_confusion_matrix=True):
    """
    Evaluate the trained SimpleCNN model on MNIST test dataset.
    
    Args:
        model_path: Path to the saved model checkpoint
        batch_size: Batch size for DataLoader
        device: Device to evaluate on (cuda/cpu). Auto-detected if None.
        show_confusion_matrix: Whether to display confusion matrix
        
    Returns:
        test_accuracy: Overall test accuracy
        all_predictions: All predictions from the model
        all_labels: All ground truth labels
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST test dataset...")
    _, test_loader = get_mnist_loaders(batch_size=batch_size, num_workers=0)
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    
    # Load trained weights
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    # Evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            accuracy = 100 * correct / total
            pbar.set_postfix({'acc': f'{accuracy:.2f}%'})
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    print(f"\n{'='*50}")
    print(f"Test Results:")
    print(f"{'='*50}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*50}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=[str(i) for i in range(10)]))
    
    # Confusion matrix
    if show_confusion_matrix:
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.2f}%')
        plt.tight_layout()
        
        # Save confusion matrix
        output_dir = Path('mnist_classification/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
        plt.show()
    
    return test_accuracy, all_predictions, all_labels


def load_model_for_inference(model_path='mnist_classification/models/mnist_cnn.pth', device=None):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load model on (cuda/cpu). Auto-detected if None.
        
    Returns:
        model: Loaded model in eval mode
        device: Device the model is on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device


if __name__ == '__main__':
    # Evaluate the trained model
    evaluate_model(
        model_path='mnist_classification/models/mnist_cnn.pth',
        batch_size=64,
        show_confusion_matrix=True
    )
