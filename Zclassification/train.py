 
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

def learn(path_to_in_domain: str, path_to_out_domain: str):
    # data_dir = f'data/{path_to_in_domain}'
    data_dir = f'Zclassification\data\in-domain-train'
    eval_dir = f'Zclassification\data\in-domain-eval'
    print(data_dir)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(data_dir, transform=transform)
    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    
    eval_data = datasets.ImageFolder(eval_dir, transform=transform)
    eval_data_loader = DataLoader(eval_data, batch_size=32, shuffle=False)

    model = myModel.SimpleCNN()

    running_loss = 0.
    last_loss = 0.

    optimizer = Adam(model.parameters(), lr=0.01) # TODO params
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i, data in enumerate(train_data_loader):
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

        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss / 1000  # loss per batch
        print(f'  batch {i+1} loss: {last_loss}')
        print('last loss:', last_loss)
        running_loss = 0.
    
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

    


learn('in-domain-train', 'out-domain-train') 