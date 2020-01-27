import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
                            
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(32*8*8, num_classes)
        
    def forward(self, x):
        print("Input shape: %s" %str(x.shape))
        out = self.layer1(x)
        print("After layer1: %s" %str(out.shape))
        out = self.layer2(out)
        print("After layer2: %s" %str(out.shape))
        out = out.reshape(out.size(0), -1)
        print("After reshape: %s" %str(out.shape))
        out = self.fc(out)
        print("After FC layer: %s" %str(out.shape))
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    print(images.shape)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    print(loss.item())
    break
