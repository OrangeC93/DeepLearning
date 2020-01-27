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
                                           download=True)
print("====================RAW DATA(Before Transform)====================")
img, label = train_dataset[0]
print(train_dataset)
print ("image format:%s, image size:%s, image mode:%s, label:%s"
    %(img.format, img.size, img.mode, label))
# img.show()

print("====================RAW DATA(After Transform)====================")
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
img, label = train_dataset[0]
print("image shape:%s, label:%s"
    %(img.shape, label))
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
                                        
print("====================DataLoader====================")
print("train set length:%s" %len(train_loader))
print("test set length:%s" %len(test_loader))
for i, (images, labels) in enumerate(train_loader):
    b, c, h, w = images.size()
    print("batch size:%s, channel:%s, height:%s, weight:%s"
        %(b, c, h, w))
    images = images.view(b, h, w) #reshape
    print(images.shape)
    plt.imshow(images[0].numpy())
    transpose_image = torch.transpose(images, 1, 0)
    print(transpose_image.shape)
    plt.imshow(transpose_image[0].numpy())
    print(labels[0])
    break