import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data/',
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
# 1. official document https://pytorch.org/docs/stable/nn.html
# 2. con2d(in_channel, out_channel, k, s, p)
#       - in channel: determined by the input itself (Grey:1, RGB:3 ... )
#       - out channel: determined by you, the lower the number, the less features this layer is gonna to learn, 
#                      as the layers get deeper, the number should increase to learn more sophisticated features.
#       - padding: should be equal to the kernel size minus 1 divided by 2,
#                  for a kernel size of 3, we would have a padding of 1.
#                  for a kernel size of 5, we would have a padding of 2.
#       - output: (height + 2 * padding - dilation * (kernel_size-1) - 1)/stride + 1 (same as weight)
# 3. example:
#       - Con2d: since 28*28 is the input, (28 + 2*2 - 1*(5-1) - 1)/1 + 1 = 28 (output width and length)
#       - Batch, ReLU 
#       - MaxPool2d: since 28*28 is the 'output',
#                    (28 + 2*0 - 1*(2-1) -1)/2 + 1 = 14 (output width and length, out_channel is determined from the begining 16)
#       - Result: 100, 16, 14, 14
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
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
        
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    print(loss.item())
    break