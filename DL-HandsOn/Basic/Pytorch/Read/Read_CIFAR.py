# https://zhuanlan.zhihu.com/p/27434001
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# torchvision dataset
cifarSet = torchvision.datasets.CIFAR10(root = "../data/cifar/", 
                                        train= True, 
                                        download = True)
print(len(cifarSet))
print(cifarSet[0])
img, label = cifarSet[0]
print (img, label)
print (img.format, img.size, img.mode)
# img.show()

mytransform = transforms.Compose([
    transforms.ToTensor()
    ]
)

# torch.utils.data.DataLoader
cifarSet = torchvision.datasets.CIFAR10(root = "../data/cifar/", 
                                        train= True, 
                                        download = True, 
                                        transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifarSet, 
                                           batch_size= 10, 
                                           shuffle= False, 
                                           num_workers= 2)
# train_loader: number of batch: 5000, batch size: 10
i = 0
for i, data in enumerate(train_loader, 0):
    # PIL
    print(data[i][5][0])
    print(data[i][5][1])
    img = transforms.ToPILImage()(data[i][0])
    # img.show()
    i = i+1
    break
print(i)