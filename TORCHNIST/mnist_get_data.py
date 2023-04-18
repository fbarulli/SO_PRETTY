import torch
from torch.utils import data
from torchvision import datasets, transforms
import os




def get_mnist_loader(batch_size : int):

    root = "./data"
    
    if not os.path.exists(root):
        os.mkdir(root)

    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root=root, download=True, train=True,transform=trans)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_data = datasets.MNIST(root=root, download=True, train=False, transform=trans)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return train_loader, test_loader
get_mnist_loader(10)