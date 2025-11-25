 
import sklearn as sk
import numpy as np
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.optim import Adam
from torch import nn
import myModel as myModel
from tqdm import tqdm

def learn(path_to_in_domain: str, path_to_out_domain: str, num_epochs=10, batch_size=32):
    # data_dir = f'data/{path_to_in_domain}'
    data_dir = f'Zclassification\data\in-domain-train'
    eval_dir = f'Zclassification\data\in-domain-eval'
    print(data_dir)
    
    # First, load data without normalization to calculate mean and std
    transform_temp = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    temp_data = datasets.ImageFolder(data_dir, transform=transform_temp)
    temp_loader = DataLoader(temp_data, batch_size=batch_size, shuffle=False)
    
    # Calculate mean and std from training data
    print("Calculating mean and std from training data...")
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in tqdm(temp_loader, desc="Computing statistics"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    print(f"Training data - Mean: {mean.item():.4f}, Std: {std.item():.4f}")
    
    # Now create transform with calculated mean and std
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean.item()], std=[std.item()])
    ])

    train_data = datasets.ImageFolder(data_dir, transform=transform)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    eval_data = datasets.ImageFolder(eval_dir, transform=transform)
    eval_data_loader = DataLoader(eval_data, batch_size=32, shuffle=False)
    
    # Get number of classes for one-hot encoding
    num_classes = len(train_data.classes)
    print(f"Number of classes: {num_classes}")

    model = myModel.AdvancedCNN(num_classes=num_classes)

    optimizer = Adam(model.parameters(), lr=0.01) # TODO params
    criterion = nn.CrossEntropyLoss()
    
    # Track loss history for plotting
    loss_history = []
    
    # Training loop with epochs
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(pbar):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = 100 * correct / total
        loss_history.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Plot training loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('Zclassification/training_loss.png', dpi=150)
    print(f"\nTraining loss plot saved to Zclassification/training_loss.png")
    plt.show()
    
    # Evaluation on in-domain eval dataset
    print("\n" + "="*50)
    print("Evaluating on in-domain eval dataset...")
    print("="*50)
    
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0
    
    with torch.no_grad():
        for data in eval_data_loader:
            inputs, labels = data
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_eval_loss = eval_loss / len(eval_data_loader)
    accuracy = 100 * correct / total
    
    print(f"\nEvaluation Results:")
    print(f"Loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)

    


learn('in-domain-train', 'out-domain-train', num_epochs=50, batch_size=32) 