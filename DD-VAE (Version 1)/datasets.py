# pytorch
import torch
import torch.utils.data as  data

# torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# paths
DATASET_PATH = './data'

# seed
seed = 7
torch.manual_seed(seed)



def get_mnist(dataloader):
    # transformations applied to all images
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # load train dataset
    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    train_set, val_set = data.random_split(train_dataset, [50000, 10000])

    # load test dataset
    test_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)

    # data loaders
    train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False)

    if dataloader == 'test': 
        return test_loader
    elif dataloader == 'val': 
        return val_loader
    else:
        return train_loader