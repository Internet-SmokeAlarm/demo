from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class CIFAR10Dataset(Dataset):

    CIFAR10_DATA_PATH = '~/Desktop/test2/data'

    def __init__(self, logger):
        super(CIFAR10Dataset, self).__init__(logger)

    def load_train_dataset(self):
        self.get_logger().debug("Loading CIFAR10 train data")

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10(root=CIFAR10Dataset.CIFAR10_DATA_PATH, train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_logger().debug("Finished loading CIFAR10 train data")

        return train_data

    def load_test_dataset(self):
        self.get_logger().debug("Loading CIFAR10 test data")

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_dataset = datasets.CIFAR10(root=CIFAR10Dataset.CIFAR10_DATA_PATH, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_logger().debug("Finished loading CIFAR10 test data")

        return test_data
