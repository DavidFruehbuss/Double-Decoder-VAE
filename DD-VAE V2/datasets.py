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

import h5py
import os
import numpy as np
from PIL import Image
import urllib.request


class fixedMNIST(data.Dataset):
    """ Binarized MNIST dataset, proposed in
    http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf """

    train_file = "binarized_mnist_train.amat"
    val_file = "binarized_mnist_valid.amat"
    test_file = "binarized_mnist_test.amat"

    def __init__(self, root, train=True, transform=None, download=False):
        # we ignore transform.
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        self.data = self._get_data(train=train)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.FloatTensor)
        return img, torch.tensor(-1)  # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):
        with h5py.File(os.path.join(self.root, "data.h5"), "r") as hf:
            data = hf.get("train" if train else "test")
            data = np.array(data)
        return data

    def get_mean_img(self):
        return self.data.mean(0).flatten()

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print("Downloading MNIST with fixed binarization...")
        for dataset in ["train", "valid", "test"]:
            filename = "binarized_mnist_{}.amat".format(dataset)
            url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
                dataset
            )
            print("Downloading from {}...".format(url))
            local_filename = os.path.join(self.root, filename)
            urllib.request.urlretrieve(url, local_filename)
            print("Saved to {}".format(local_filename))

        def filename_to_np(filename):
            with open(filename) as f:
                lines = f.readlines()
            return np.array([[int(i) for i in line.split()] for line in lines]).astype(
                "int8"
            )

        train_data = np.concatenate(
            [
                filename_to_np(os.path.join(self.root, self.train_file)),
                filename_to_np(os.path.join(self.root, self.val_file)),
            ]
        )
        test_data = filename_to_np(os.path.join(self.root, self.val_file))
        with h5py.File(os.path.join(self.root, "data.h5"), "w") as hf:
            hf.create_dataset("train", data=train_data.reshape(-1, 28, 28))
            hf.create_dataset("test", data=test_data.reshape(-1, 28, 28))
        print("Done!")

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "data.h5"))

def get_mnist(dataloader):
    # transformations applied to all images
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # binarized mnist
    loader_fn, root = fixedMNIST, DATASET_PATH + "/fixedmnist"
    
    # load train dataset
    train_dataset = loader_fn(root=root, train=True, download=True, transform=transform)
    train_set, val_set = data.random_split(train_dataset, [50000, 10000])

    # load test dataset
    test_set = loader_fn(root=root, train=False, download=True, transform=transform)

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