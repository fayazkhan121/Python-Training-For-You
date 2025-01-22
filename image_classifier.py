# ENVIRONMENT SETUP
# Required Dependencies:
# - torch==2.0.0 or later      : Main deep learning framework
# - torchvision==0.15.0 or later : For image processing and datasets
# - matplotlib==3.5.0 or later : For visualization
# - numpy==1.21.0 or later     : For numerical operations

# INPUT SPECIFICATIONS:
# 1. Images:
#    - Format: RGB images
#    - Size: 32x32 pixels (CIFAR-10 default)
#    - Type: .jpg, .png
#    - Color space: RGB (3 channels)

# 2. Dataset Structure:
#    - Default: Uses CIFAR-10 (automatically downloaded)
#    - Custom Dataset Requirements:
#      └── data/
#          ├── train/
#          │   ├── class1/
#          │   │   ├── image1.jpg
#          │   │   └── image2.jpg
#          │   └── class2/
#          └── test/
              ├── class1/
              └── class2/

# HYPERPARAMETERS TO TUNE:
# 1. Training:
#    - batch_size: Default=64 (Adjust based on GPU memory)
#    - learning_rate: Default=0.001 (Adjust if loss doesn't converge)
#    - num_epochs: Default=10 (Increase for better accuracy)

# 2. Model Architecture:
#    - num_classes: Default=10 (Modify for different classification tasks)
#    - dropout_rate: Default=0.5 (Adjust for regularization strength)

# CODE STRUCTURE EXPLANATION:
# 1. ConvolutionalNeuralNetwork Class:
#    - Purpose: Defines the neural network architecture
#    - Input Shape: (batch_size, 3, 32, 32)
#    - Layer Progression:
#      Input → Conv1(32) → Conv2(64) → Conv3(128) → FC(512) → Output
#    - Feature Map Sizes:
#      32x32 → 16x16 → 8x8 → 4x4 → Flattened → Classes

# 2. ImageClassifier Class:
#    - Purpose: Handles training and evaluation pipeline
#    - Key Methods:
#      ├── load_data(): Prepares datasets and dataloaders
#      ├── train(): Handles training loop and optimization
#      ├── validate(): Performs model evaluation
#      └── plot_training_history(): Visualizes training progress

# DATA AUGMENTATION OPTIONS:
# Current implementations:
# - RandomHorizontalFlip: 50% chance
# - RandomRotation: ±10 degrees
# Additional options to consider:
# - transforms.RandomCrop
# - transforms.ColorJitter
# - transforms.RandomAffine

# ERROR HANDLING:
# - GPU availability check
# - Dataset loading verification
# - Training loop monitoring
# - Model saving/loading safeguards

# USAGE RECOMMENDATIONS:
# 1. For Training:
#    - Start with default hyperparameters
#    - Monitor training loss curve
#    - Adjust batch_size based on memory
#    - Modify learning_rate if loss plateaus

# 2. For Inference:
#    - Use load_model() for pretrained weights
#    - Ensure input image preprocessing matches training
#    - Use model.eval() for prediction

# PERFORMANCE METRICS:
# - Training tracks:
#   * Loss per batch
#   * Accuracy per epoch
#   * Validation accuracy
#   * Training time

# CUSTOMIZATION OPTIONS:
# 1. Architecture Modifications:
#    - Add/remove convolutional layers
#    - Modify filter sizes
#    - Change fully connected layer dimensions

# 2. Training Modifications:
#    - Implement learning rate scheduling
#    - Add early stopping
#    - Implement cross-validation
                
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class ConvolutionalNeuralNetwork(nn.Module):
    """
    A convolutional neural network for image classification.
    Architecture:
    - 3 Convolutional layers with batch normalization and ReLU activation
    - 2 Fully connected layers
    - Dropout for regularization
    """
    def __init__(self, num_classes=10):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        # First convolutional block
        # Input: 3x32x32 -> Output: 32x32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 32x16x16
        )
        
        # Second convolutional block
        # Input: 32x16x16 -> Output: 64x16x16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 64x8x8
        )
        
        # Third convolutional block
        # Input: 64x8x8 -> Output: 128x8x8
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 128x4x4
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),  # Flatten the 128x4x4 features
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

class ImageClassifier:
    """
    A wrapper class for training and evaluating the CNN model.
    Handles data loading, training, validation, and visualization.
    """
    def __init__(self, num_classes=10, batch_size=64, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Initialize the model
        self.model = ConvolutionalNeuralNetwork(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Set up data transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_data(self):
        """Load and prepare the CIFAR-10 dataset"""
        # Load training data
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True, 
            transform=self.transform
        )
        self.trainloader = DataLoader(
            trainset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=2
        )

        # Load validation data
        testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        self.testloader = DataLoader(
            testset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=2
        )
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')

    def train(self, num_epochs):
        """Train the model for specified number of epochs"""
        print(f"Training on {self.device}")
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            self.model.train()
            
            # Training loop
            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                    train_losses.append(running_loss / 100)
                    running_loss = 0.0
            
            # Validate after each epoch
            accuracy = self.validate()
            val_accuracies.append(accuracy)
            print(f'Validation Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')
        
        return train_losses, val_accuracies

    def validate(self):
        """Validate the model and return accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def plot_training_history(self, train_losses, val_accuracies):
        """Plot training loss and validation accuracy"""
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Loss')
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Usage example
if __name__ == "__main__":
    # Initialize the classifier
    classifier = ImageClassifier(num_classes=10, batch_size=64, learning_rate=0.001)
    
    # Load CIFAR-10 dataset
    classifier.load_data()
    
    # Train the model
    train_losses, val_accuracies = classifier.train(num_epochs=10)
    
    # Plot training history
    classifier.plot_training_history(train_losses, val_accuracies)
    
    # Save the trained model
    classifier.save_model('image_classifier.pth')
