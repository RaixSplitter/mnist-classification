import torch
import matplotlib.pyplot as plt
import random
from pathlib import Path

from data import get_mnist_loaders
from my_models import SimpleCNN


def show_random_predictions(model_path='mnist_classification/models/mnist_cnn.pth',
                           num_samples=9,
                           device=None):
    """
    Display random MNIST images with their predictions.
    
    Args:
        model_path: Path to the saved model checkpoint
        num_samples: Number of random samples to display (default: 9 for 3x3 grid)
        device: Device to run inference on (cuda/cpu). Auto-detected if None.
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleCNN(num_classes=10)
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load test data
    _, test_loader = get_mnist_loaders(batch_size=1, num_workers=0)
    
    # Get all test samples
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    
    # Select random samples
    total_samples = len(all_images)
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Prepare grid
    grid_size = int(num_samples ** 0.5)
    if grid_size * grid_size < num_samples:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    # Make predictions and display
    with torch.no_grad():
        for idx, sample_idx in enumerate(random_indices):
            if idx >= len(axes):
                break
                
            image = all_images[sample_idx].to(device)
            true_label = all_labels[sample_idx].item()
            
            # Get prediction
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()
            
            # Get confidence (softmax probability)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0][predicted_label].item() * 100
            
            # Display image
            img = image.cpu().squeeze().numpy()
            axes[idx].imshow(img, cmap='gray')
            axes[idx].axis('off')
            
            # Color code: green if correct, red if wrong
            color = 'green' if predicted_label == true_label else 'red'
            title = f'Pred: {predicted_label} ({confidence:.1f}%)\nTrue: {true_label}'
            axes[idx].set_title(title, color=color, fontsize=10, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=1.5)
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Random MNIST Predictions', fontsize=16, fontweight='bold')
    
    # Save figure
    output_dir = Path('mnist_classification/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'random_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Predictions saved to {output_path}")
    
    plt.show()


def predict_single_image(model_path='mnist_classification/models/mnist_cnn.pth', 
                        image_index=None,
                        device=None):
    """
    Make a prediction on a single random or specified image.
    
    Args:
        model_path: Path to the saved model checkpoint
        image_index: Specific index to predict (None for random)
        device: Device to run inference on (cuda/cpu). Auto-detected if None.
        
    Returns:
        predicted_label: The predicted digit
        true_label: The actual digit
        confidence: Prediction confidence percentage
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimpleCNN(num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    _, test_loader = get_mnist_loaders(batch_size=1, num_workers=0)
    
    # Get specific or random image
    if image_index is None:
        image_index = random.randint(0, len(test_loader) - 1)
    
    for idx, (images, labels) in enumerate(test_loader):
        if idx == image_index:
            image = images.to(device)
            true_label = labels.item()
            break
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = predicted.item()
        confidence = confidence.item() * 100
    
    # Display result
    plt.figure(figsize=(12, 12))
    img = image.cpu().squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    color = 'green' if predicted_label == true_label else 'red'
    title = f'Prediction: {predicted_label} ({confidence:.2f}%)\nTrue Label: {true_label}'
    plt.title(title, color=color, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return predicted_label, true_label, confidence


if __name__ == '__main__':
    # Show 9 random predictions in a grid
    show_random_predictions(
        model_path='mnist_classification/models/mnist_cnn.pth',
        num_samples=9
    )
    
    # Optionally show a single random prediction
    # predict_single_image(model_path='mnist_classification/models/mnist_cnn.pth')
