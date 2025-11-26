################################################################################
# evaluationResNet.py
#
# Evaluation script for fine-tuned ResNet50 model on image classification task.
#
# This script:
# - Loads a pre-trained and fine-tuned ResNet50 model from checkpoint
# - Evaluates the model on validation and test datasets
# - Generates comprehensive evaluation metrics including:
#   * Accuracy and loss on validation and test sets
#   * Confusion matrix visualization
#   * Per-class precision, recall, and F1-scores
# - Logs all results to a dedicated log file
#
# Usage:
#   python evaluationResNet.py
#
# Requirements:
#   - Trained model checkpoint at MODEL_PATH
#   - Data loaders configured in dataResNet.py
#   - All dependencies: torch, torchvision, sklearn, seaborn, matplotlib
################################################################################

"""
Evaluation script for ResNet50 fine-tuned model
"""

from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from dataResNet import get_data_loaders

# ========================== CONFIG ==========================
MODEL_PATH = 'Zclassification/models/resnet50_finetuned.pth'
BATCH_SIZE = 32
CONFUSION_MATRIX_PATH = 'Zclassification/plots/confusion_matrix_resnet50.png'
LOG_FILE = 'Zclassification/logs/eval_resnet.log'
# ============================================================

# Create directories
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


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load the fine-tuned ResNet50 model from checkpoint.
    
    Reconstructs the model architecture and loads saved weights:
    - Creates ResNet50 architecture matching the saved checkpoint
    - Loads model state dict from checkpoint file
    - Moves model to specified device
    - Sets model to evaluation mode
    - Extracts and logs metadata from checkpoint
    
    Args:
        model_path (str): Path to the saved model checkpoint (.pth file)
        device (torch.device): Device to load model on (cuda or cpu)
    
    Returns:
        tuple containing:
            - model (nn.Module): Loaded ResNet50 model in eval mode
            - checkpoint (dict): Full checkpoint dictionary containing:
                * 'model_state_dict': Model weights
                * 'epoch': Number of epochs trained
                * 'num_classes': Number of output classes
                * 'class_names': List of class names
                * Additional training metadata
    
    Raises:
        FileNotFoundError: If model_path does not exist
        KeyError: If checkpoint is missing required keys
        RuntimeError: If model architecture doesn't match saved weights
    """
    logger.info(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    
    # Create model architecture
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Trained for {checkpoint['epoch']} epochs")
    logger.info(f"Number of classes: {num_classes}")
    
    return model, checkpoint


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device, dataset_name: str = "Test") -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate model on a dataset.
    
    Comprehensive evaluation that:
    - Runs inference on entire dataset without gradient computation
    - Computes loss and accuracy metrics
    - Collects all predictions and true labels for further analysis
    - Displays progress bar with real-time metrics
    
    Args:
        model (nn.Module): Model to evaluate (should be in eval mode)
        data_loader (DataLoader): DataLoader for the evaluation dataset
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        device (torch.device): Device to run evaluation on (cuda or cpu)
        dataset_name (str, optional): Name of dataset for logging. Defaults to "Test".
    
    Returns:
        tuple containing:
            - loss (float): Average loss across all batches
            - accuracy (float): Overall accuracy percentage (0-100)
            - all_predictions (list[int]): List of predicted class indices for all samples
            - all_labels (list[int]): List of true class indices for all samples
    
    Note:
        The predictions and labels can be used for:
        - Confusion matrix generation
        - Per-class precision/recall/F1 calculation
        - Error analysis on misclassified samples
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    logger.info(f"\nEvaluating on {dataset_name} dataset...")
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"{dataset_name} Evaluation")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0
            accuracy = 100 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], accuracy: float) -> None:
    """
    Plot and save confusion matrix visualization.
    
    Creates a heatmap showing the confusion matrix:
    - Rows represent true labels
    - Columns represent predicted labels
    - Cell values show count of samples
    - Diagonal shows correct predictions
    - Off-diagonal shows misclassifications
    
    Args:
        y_true (list[int]): True class labels (ground truth)
        y_pred (list[int]): Predicted class labels from model
        class_names (list[str]): List of class names for axis labels
        accuracy (float): Overall accuracy percentage to display in title
    
    Returns:
        None
    
    Side Effects:
        - Saves confusion matrix plot to CONFUSION_MATRIX_PATH
        - Displays plot in GUI window
        - Logs save location to logger
    
    Note:
        The confusion matrix helps identify:
        - Which classes are frequently confused
        - Class-specific performance issues
        - Systematic errors in the model
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - Test Accuracy: {accuracy:.2f}%',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    logger.info(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    plt.show()


def print_classification_report(y_true: List[int], y_pred: List[int], 
                               class_names: List[str]) -> None:
    """
    Print detailed classification report with per-class metrics.
    
    Generates and logs a comprehensive report including:
    - Per-class precision: True positives / (True positives + False positives)
    - Per-class recall: True positives / (True positives + False negatives)
    - Per-class F1-score: Harmonic mean of precision and recall
    - Support: Number of samples per class
    - Overall macro and weighted averages
    
    Args:
        y_true (list[int]): True class labels (ground truth)
        y_pred (list[int]): Predicted class labels from model
        class_names (list[str]): List of class names for labeling
    
    Returns:
        None
    
    Side Effects:
        - Logs detailed classification report to logger
        - Output is also saved to LOG_FILE
    
    Note:
        Use this report to identify:
        - Classes with low precision (many false positives)
        - Classes with low recall (many false negatives)
        - Overall model balance across classes
    """
    logger.info("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\n" + report)


def run_evaluation(model_path: str = MODEL_PATH, batch_size: int = BATCH_SIZE) -> Dict:
    """
    Main evaluation function - comprehensive model assessment.
    
    Complete evaluation pipeline that:
    - Loads trained model from checkpoint
    - Loads validation and test data
    - Evaluates model on both datasets
    - Generates confusion matrix visualization
    - Prints detailed classification report
    - Logs comprehensive evaluation summary
    
    Args:
        model_path (str, optional): Path to model checkpoint. Defaults to MODEL_PATH.
        batch_size (int, optional): Batch size for evaluation. Defaults to BATCH_SIZE.
    
    Returns:
        dict: Dictionary containing evaluation results:
            - 'val_loss' (float): Validation set loss
            - 'val_acc' (float): Validation set accuracy (%)
            - 'test_loss' (float): Test set loss
            - 'test_acc' (float): Test set accuracy (%)
            - 'test_preds' (list[int]): All test predictions
            - 'test_labels' (list[int]): All test true labels
    
    Side Effects:
        - Saves confusion matrix to CONFUSION_MATRIX_PATH
        - Logs all results to LOG_FILE
        - Displays plots and progress bars
    
    Raises:
        FileNotFoundError: If model checkpoint doesn't exist
        RuntimeError: If data directories are not accessible
    
    Note:
        This function provides the most comprehensive evaluation.
        Use after training to assess final model performance.
    """
    logger.info("="*60)
    logger.info("Starting ResNet50 Model Evaluation")
    logger.info("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(batch_size)
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    
    # Setup criterion
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on validation set
    logger.info("\n" + "="*60)
    val_loss, val_acc, val_preds, val_labels = evaluate_model(
        model, val_loader, criterion, device, "Validation"
    )
    logger.info(f"Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    test_loss, test_acc, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion, device, "Test"
    )
    logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # Print classification report
    print_classification_report(test_labels, test_preds, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, class_names, test_acc)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    logger.info(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    logger.info("="*60)
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_preds': test_preds,
        'test_labels': test_labels
    }


if __name__ == '__main__':
    run_evaluation()
